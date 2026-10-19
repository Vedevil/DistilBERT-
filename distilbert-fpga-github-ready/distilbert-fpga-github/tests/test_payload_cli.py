from pathlib import Path
import json,subprocess,sys,hashlib
import numpy as np
from reference.toy_model import smoke_frames,evaluate
R=Path(__file__).resolve().parents[1]

def call(*args):return subprocess.run([sys.executable,str(R/'host/step16/prepare_payload.py'),*map(str,args)],text=True,capture_output=True)
def test_toy_payload_cli_is_self_contained(tmp_path):
    out=tmp_path/'payload';p=call('--toy','--out-dir',out);assert p.returncode==0,p.stderr
    info=json.loads(p.stdout);assert info['synthetic_toy'] and info['words']==6110 and info['bytes']==24440
    raw=(out/'parameters.bin').read_bytes();m=json.loads((out/'parameter_layout.json').read_text());assert hashlib.sha256(raw).hexdigest()==m['logical_signed32_SHA256']
    known=json.loads((out/'known_cases.json').read_text())['synthetic_toy'];assert known['logits_q11']==[-71592,-524367] and known['mask']==[1,0]
    assert np.load(out/'embeddings_q11.npy',allow_pickle=False).dtype==np.int32

def test_payload_existing_output_refused(tmp_path):
    out=tmp_path/'payload';out.mkdir();(out/'user.txt').write_text('keep');p=call('--toy','--out-dir',out);assert p.returncode!=0 and (out/'user.txt').read_text()=='keep'

def test_canonical_external_model_export_matches_toy(tmp_path):
    x,ps,_,h=smoke_frames();model=tmp_path/'model';model.mkdir()
    for i,p in enumerate(ps):np.savez(model/f'encoder{i}.npz',**p)
    np.savez(model/'classifier.npz',**h);np.save(tmp_path/'input.npy',x,allow_pickle=False)
    p=call('--model-dir',model,'--embeddings',tmp_path/'input.npy','--mask','10','--out-dir',tmp_path/'out');assert p.returncode==0,p.stderr
    assert not json.loads(p.stdout)['synthetic_toy'] and json.loads((tmp_path/'out/mask.json').read_text())==[1,0]

def test_external_export_requires_input_and_mask(tmp_path):
    p=call('--model-dir',tmp_path,'--out-dir',tmp_path/'out');assert p.returncode!=0 and 'requires --embeddings and --mask' in p.stderr

def test_payload_mask_shape_rejected(tmp_path):
    p=call('--toy','--mask','111','--out-dir',tmp_path/'out');assert p.returncode!=0 and not (tmp_path/'out').exists()

def test_independent_model_toy_expected_logits():
    x,p,m,h=smoke_frames();g=evaluate(x,p,h,m[0]);assert g['logits'].tolist()==[-71592,-524367] and g['prediction']==0
