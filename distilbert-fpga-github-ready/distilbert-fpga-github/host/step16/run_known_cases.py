"""BOARD-ONLY entry point; not executed on ZCU104 by the assistant."""
import argparse,json
from pathlib import Path
import numpy as np
from pynq_driver import DistilBert
p=argparse.ArgumentParser();p.add_argument('--bit',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--ip-name',default='distilbert_0');p.add_argument('--timeout',type=float,default=300);a=p.parse_args()
d=DistilBert.from_overlay(a.bit,a.ip_name)
# Do NOT free or reload overlay on an unquiesced bus error; driver pins ownership.
try:
    d.load_parameters(a.payload/'parameters.bin',a.payload/'parameter_layout.json')
    x=np.load(a.payload/'embeddings_q11.npy',allow_pickle=False)
    for name,k in json.loads((a.payload/'known_cases.json').read_text()).items():
        got=d.infer(x,k['mask'],timeout=a.timeout);assert got['logits_q11']==k['logits_q11'] and got['prediction']==k['prediction'],(name,got,k)
        print(name,json.dumps(got))
except BaseException:
    if d._active:
        # A global Python pin does NOT survive process/kernel termination.
        # Keep this process alive until drain succeeds or platform is reset.
        import time
        print('UNQUIESCED AXI: KEEPING PROCESS/BUFFERS ALIVE. Do not kill/restart kernel or reload overlay. Resolve bus fault or coordinate FULL platform reset.',flush=True)
        while d._active:
            try:d.abort(timeout=5)
            except BaseException as recovery:
                print('Still pinned:',str(recovery),flush=True)
                try:time.sleep(1)
                except BaseException:pass
    raise
else:d.close()
