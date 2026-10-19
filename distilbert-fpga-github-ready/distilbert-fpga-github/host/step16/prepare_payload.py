"""Strict portable DDR exporter. No trained weights or old workspace are required.
Use --toy for synthetic testing, or supply six canonical encoder NPZ + head NPZ.
The wire format is kept compatible with the verified Step16 hardware/driver.
"""
from pathlib import Path
import sys,json,argparse,hashlib,tempfile
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from reference.parameter_image import pack,ENC_KEYS,HEAD_KEYS

def convert(text,path,manifest):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    count=0;h=hashlib.sha256()
    with Path(text).open() as src,path.open('xb') as dst:
        while True:
            words=[]
            for _ in range(65536):
                w=src.readline()
                if not w:break
                words.append(int(w.strip(),16))
            if not words:break
            raw=np.asarray(words,dtype='<u4').tobytes();dst.write(raw);h.update(raw);count+=len(words)
    if count!=manifest['total_words'] or h.hexdigest()!=manifest['logical_signed32_SHA256']:raise ValueError('binary conversion mismatch')
    return {'words':count,'bytes':path.stat().st_size,'SHA256':h.hexdigest()}

def load_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('unexpected tensor keys in '+str(path))
        return {k:z[k].copy() for k in keys}

def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--toy',action='store_true');g.add_argument('--model-dir',type=Path)
    ap.add_argument('--embeddings',type=Path);ap.add_argument('--mask',help='N binary key-keep digits, e.g. 1110');ap.add_argument('--out-dir',type=Path,default=R/'.cache/payload');a=ap.parse_args()
    if a.toy:
        from reference.toy_model import smoke_frames
        x,p,m,hp=smoke_frames();mask=m[0]
    else:
        if a.embeddings is None or a.mask is None:ap.error('--model-dir requires --embeddings and --mask')
        p=[load_npz(a.model_dir/f'encoder{i}.npz',ENC_KEYS) for i in range(6)];hp=load_npz(a.model_dir/'classifier.npz',HEAD_KEYS);x=np.load(a.embeddings,allow_pickle=False);mask=None
    if a.mask is not None:
        if any(v not in '01' for v in a.mask):ap.error('--mask must contain only 0 or 1')
        mask=np.asarray([int(v) for v in a.mask],np.uint8)
    if x.dtype.kind!='i' or x.dtype.itemsize!=4 or x.dtype.byteorder=='>':raise ValueError('embeddings must be signed32 little/native-endian Q11; no implicit cast/clipping')
    if x.ndim!=2 or not 1<=len(x)<=16 or x.shape[1]!=len(p[0]['out_bias']) or np.shape(mask)!=(len(x),):raise ValueError('embedding/mask dimensions; default wrapper supports N1..16')
    a.out_dir.mkdir(parents=True,exist_ok=False);cache=R/'.cache';cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pack_',dir=cache) as tmp:
        manifest=pack(Path(tmp)/'parameters.mem',p,hp);result=convert(Path(tmp)/'parameters.mem',a.out_dir/'parameters.bin',manifest)
    (a.out_dir/'parameter_layout.json').write_text(json.dumps(manifest,indent=2)+'\n');np.save(a.out_dir/'embeddings_q11.npy',x,allow_pickle=False);(a.out_dir/'mask.json').write_text(json.dumps(mask.tolist())+'\n')
    if a.toy:
        from reference.toy_model import evaluate
        gold=evaluate(x,p,hp,mask)
        (a.out_dir/'known_cases.json').write_text(json.dumps({'synthetic_toy':{'mask':mask.tolist(),'logits_q11':gold['logits'].tolist(),'prediction':gold['prediction']}},indent=2)+'\n')
    print(json.dumps({**result,'synthetic_toy':a.toy,'dimensions':{'dim':manifest['dim'],'heads':manifest['heads'],'ff_hidden':manifest['ff_hidden'],'head_hidden':manifest['head_hidden']},'note':'Toy payload is NOT compatible with the default DIM768 board configuration.' if a.toy else 'Trained weights were supplied externally; not included in this repository.'},indent=2))
if __name__=='__main__':main()
