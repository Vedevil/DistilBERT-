#!/usr/bin/env python3
"""Optional: convert YOUR separate full Step16 workspace to canonical model NPZ.
No old workspace/model data is shipped with this code-only repository.
"""
from pathlib import Path
import argparse,subprocess,sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True,help='path containing old distilbert-fpga-attention/');p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();project=a.workspace.resolve()/'distilbert-fpga-attention';out=a.out_dir.resolve()
if not (project/'reference/zcu104_step16_inputs.py').is_file():p.error('not a full Step16 workspace')
if out.exists():p.error('output directory must be NEW')
code='''import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from reference.zcu104_step16_inputs import actual
x,ps,ms,hp=actual('unmasked');out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
for i,p in enumerate(ps):np.savez_compressed(out/f'encoder{i}.npz',**p)
np.savez_compressed(out/'classifier.npz',**hp);np.save(out/'embeddings_q11.npy',x,allow_pickle=False)
print('Exported canonical externally supplied weights; use mask111 or110 for known N3 cases.')
'''
subprocess.run([sys.executable,'-c',code,str(out)],cwd=project,check=True)
