"""BOARD ONLY: requires a real matching bit/hwh and externally supplied payload."""
from pathlib import Path
import argparse,json,time
import numpy as np
from pynq_driver import DistilBert

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bit',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--ip-name',default='distilbert_0');p.add_argument('--timeout',type=float,default=300);a=p.parse_args()
    d=DistilBert.from_overlay(a.bit,a.ip_name)
    try:
        d.load_parameters(a.payload/'parameters.bin',a.payload/'parameter_layout.json')
        x=np.load(a.payload/'embeddings_q11.npy',allow_pickle=False);mask=json.loads((a.payload/'mask.json').read_text());print(json.dumps(d.infer(x,mask,timeout=a.timeout),indent=2))
    except BaseException:
        if d._active:
            print('UNQUIESCED AXI: KEEPING PROCESS/BUFFERS ALIVE. Do not kill/restart Python or reload overlay; resolve bus fault or coordinate full platform reset.',flush=True)
            while d._active:
                try:d.abort(timeout=5)
                except BaseException as recovery:
                    print('Still pinned:',recovery,flush=True)
                    try:time.sleep(1)
                    except BaseException:pass
        raise
    else:d.close()
if __name__=='__main__':main()
