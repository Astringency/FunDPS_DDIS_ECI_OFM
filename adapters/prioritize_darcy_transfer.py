"""Temporarily defer this task's two lower-priority local uploads for Darcy.

Resume automatically when Darcy's checked transfer ends, after 30 minutes,
or on interruption. Never signal a reused PID or a process already stopped.
"""
import json
import os
from pathlib import Path
import signal
import time


ROOT=Path('/home/tat512/C01Python/DDIS_comparison_20260919')
TARGETS={653639:'transfers/extended/nsnonbounded/',655881:'transfers/shared_prior_assets/'}


def identity(pid):
    proc=Path('/proc')/str(pid)
    try:
        stat=(proc/'stat').read_text().split(') ',1)[1].split()
        argv=(proc/'cmdline').read_bytes().split(b'\0')[:-1]
        return {'pid':pid,'start_ticks':int(stat[19]),'state':stat[0],
                'argv':[part.decode() for part in argv]}
    except FileNotFoundError:
        return None


def main():
    records=[]
    for pid,suffix in TARGETS.items():
        record=identity(pid)
        assert record and record['state']!='T'
        assert Path(record['argv'][0]).name=='rsync'
        assert str(ROOT/suffix)+'/' in record['argv']
        assert any(arg.startswith('server216:') for arg in record['argv'])
        records.append(record)
    state=ROOT/'transfers/darcy-priority.json'
    journal={'reason':'Fill idle training GPU by finishing Darcy before NS and ordinary-FM weights',
             'started_unix':time.time(),'max_seconds':1800,'targets':records,'suspended':[]}
    def save():
        state.write_text(json.dumps(journal,indent=2)+'\n')
    def interrupted(sig,frame):
        raise InterruptedError(f'Received signal {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for record in records:
            current=identity(record['pid'])
            assert current and current['start_ticks']==record['start_ticks'] and current['argv']==record['argv']
            journal['suspended'].append(record['pid'])
            save()
            os.kill(record['pid'],signal.SIGSTOP)
        print('Prioritizing Darcy; lower-priority upload PIDs suspended:',journal['suspended'],flush=True)
        while not (ROOT/'transfers/extended/darcy.exit').exists() and time.time()-journal['started_unix']<1800:
            time.sleep(5)
    finally:
        journal['resumed']=[]
        for record in records:
            current=identity(record['pid'])
            if record['pid'] in journal['suspended'] and current and current['start_ticks']==record['start_ticks'] and current['argv']==record['argv']:
                os.kill(record['pid'],signal.SIGCONT)
                journal['resumed'].append(record['pid'])
        journal['finished_unix']=time.time()
        save()
        print('Restored uploads:',journal['resumed'],flush=True)


if __name__=='__main__':
    main()
