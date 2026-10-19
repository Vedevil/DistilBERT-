#!/usr/bin/env python3
"""Code-only verification runner. No trained model data or old workspace needed."""
from pathlib import Path
import argparse,subprocess,shutil,time,json,sys
ROOT=Path(__file__).resolve().parents[1]
CASES={'reader':('axi_word_reader_tb',[ROOT/'rtl/axi_word_reader_q.sv',ROOT/'tb/axi_word_reader_tb.sv']),'control':('zcu104_control_tb',[*sorted((ROOT/'rtl').glob('*.sv')),ROOT/'tb/zcu104_control_tb.sv']),'demo':('zcu104_protocol_tb',[ROOT/'examples/step16_axi_ddr_demo.sv'])}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--test',choices=['all',*CASES],default='all');p.add_argument('--out-dir',type=Path,default=ROOT/'runs/latest');a=p.parse_args()
    for tool in ['iverilog','vvp']:
        if not shutil.which(tool):p.error(f'{tool} not found. Install Icarus Verilog (Ubuntu/Debian: sudo apt install iverilog).')
    a.out_dir.mkdir(parents=True,exist_ok=False);results={}
    for name in CASES if a.test=='all' else [a.test]:
        top,sources=CASES[name];work=(a.out_dir/name).resolve();work.mkdir();image=work/'sim.vvp';t=time.monotonic();cmd=['iverilog','-g2012','-s',top,'-o',str(image),*map(str,sources)];cp=subprocess.run(cmd,text=True,capture_output=True);(work/'compile.log').write_text(cp.stdout+cp.stderr)
        if cp.returncode:raise RuntimeError('Compile failed; see '+str(work/'compile.log'))
        cp=subprocess.run(['vvp',str(image)],cwd=work,text=True,capture_output=True,timeout=120);(work/'simulation.log').write_text(cp.stdout+cp.stderr)
        if cp.returncode or 'PASS '+top not in cp.stdout:raise RuntimeError('Simulation failed; see '+str(work/'simulation.log'))
        results[name]={'pass':True,'top':top,'elapsed_host_seconds':time.monotonic()-t,'board_tested':False};print('\n'.join(line for line in cp.stdout.splitlines() if line.startswith('PASS ')),flush=True)
    (a.out_dir/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
