"""Fetch the pinned upstream implementations without changing existing checkouts."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--names',nargs='+',default=['DDIS','FunDPS','OFM','ECI','neuraloperator']);a=p.parse_args()
pins=json.loads((ROOT/'configs/official_sources.json').read_text())
for name in a.names:
    spec=pins[name];dest=ROOT/'official'/name
    if not dest.exists():
        subprocess.run(['git','clone','--no-checkout',spec['url'],str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',spec['revision']],check=True)
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=spec['revision']:raise RuntimeError(f'{name}: expected {spec["revision"]}, found {actual}')
    print(name,actual)
