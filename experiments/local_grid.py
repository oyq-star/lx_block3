"""Resume the fixed Windows one-GPU experiment grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
MODES=('clean','iid','dispersed','crop','block')
VIEWS='clean,iid,dispersed,block,unseen300,global_relation,global_time'


def save_state(path,state):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temporary.replace(path)


def invoke(path,command,seconds,env):
    path.mkdir(parents=True,exist_ok=True)
    suffix=time.strftime('%Y%m%d_%H%M%S')
    log=path/('console_'+suffix+'.log')
    print(json.dumps(dict(stage='launch',path=str(path),command=command)),flush=True)
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as stream:
        try:
            process=subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,
                                   timeout=seconds,env=env)
            code=process.returncode
        except subprocess.TimeoutExpired:
            code=124
    print(json.dumps(dict(stage='exit',path=str(path),seconds=time.monotonic()-start,code=code)),flush=True)
    return code,str(log)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',default='cadets,theia')
    parser.add_argument('--views',default=VIEWS)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/full_grid_v1')
    parser.add_argument('--seeds',default='17,23,42')
    parser.add_argument('--budget-seconds',type=int,default=43200)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    state_path=out/'grid_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
    state['status']='RUNNING';save_state(state_path,state)
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    deadline=time.monotonic()+args.budget_seconds
    for dataset in args.datasets.split(','):
        cache=ROOT/'data/cache'/(dataset+'_longoutage_v1')
        for name in ['validation']+['test_'+view for view in args.views.split(',')]:
            while not (cache/name/'CACHE_COMPLETE').exists() or not (cache/'dataset.json').exists():
                if time.monotonic()>=deadline:raise TimeoutError('Missing cache: '+str(cache/name))
                time.sleep(5)
        jobs=[]
        stats=out/dataset/'statistics'
        jobs.append((stats,['statistical_baselines.py','--cache',str(cache),'--output',str(stats),'--views',args.views],600))
        smoke=out/dataset/'magic_cuda_sanity'
        jobs.append((smoke,['magic_adapted.py','--cache',str(cache),'--output',str(smoke),'--sanity'],180))
        magic=out/dataset/'magic_adapted'
        jobs.append((magic,['magic_adapted.py','--cache',str(cache),'--output',str(magic)],1200))
        for seed in map(int,args.seeds.split(',')):
            for mode in MODES:
                folder=out/dataset/(str(seed)+'_'+mode)
                bank=cache/'train_bank_v2'/mode
                jobs.append((folder,['train_eval.py','--cache',str(cache),'--output',str(folder),'--mode',mode,
                                     '--seed',str(seed),'--epochs','3','--batch-size','64','--eval-batch-size','1024',
                                     '--train-mask-durations','1790,1799','--train-bank',str(bank),'--views',args.views,
                                     '--max-seconds','1195'],1200))
        for folder,parts,timeout in jobs:
            name=folder.relative_to(out).as_posix()
            result=folder/'results.json'
            if result.exists() and json.loads(result.read_text())['status'] in {'COMPLETED','SANITY_PASSED'}:
                state['jobs'].setdefault(name,{'status':'COMPLETED','reused':True});continue
            if folder==magic and not (smoke/'results.json').exists():
                state['jobs'][name]=dict(status='FAILED_SANITY',reason='CUDA sanity did not complete')
                save_state(state_path,state);continue
            if '--train-bank' in parts:
                bank=Path(parts[parts.index('--train-bank')+1])
                while not all((bank/(str(epoch)+'.npz')).exists() for epoch in range(3)):
                    if time.monotonic()>=deadline:raise TimeoutError('Missing train bank: '+str(bank))
                    time.sleep(5)
            seconds=min(timeout,int(deadline-time.monotonic()))
            if seconds<10:
                state['status']='BUDGET_STOP';save_state(state_path,state);return
            command=[sys.executable,str(ROOT/'experiments'/parts[0])]+parts[1:]
            state['jobs'][name]=dict(status='RUNNING',command=command);save_state(state_path,state)
            start=time.monotonic()
            code,log=invoke(folder,command,seconds,env)
            status='COMPLETED' if code==0 and result.exists() else 'FAILED'
            state['jobs'][name].update(status=status,returncode=code,wall_seconds=time.monotonic()-start,log=log)
            save_state(state_path,state)
    state['status']='COMPLETED' if all(row['status']=='COMPLETED' for row in state['jobs'].values()) else 'PARTIAL'
    save_state(state_path,state)


if __name__=='__main__':main()
