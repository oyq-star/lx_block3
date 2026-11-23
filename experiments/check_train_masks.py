"""Train-only diagnostic for the proposal's one-time duration adjustment gate."""
import json,time
from pathlib import Path
import numpy as np
from causal_data import EventDay,draw_contexts,NS

ROOT=Path(__file__).resolve().parents[1]
def main():
    started=time.monotonic();results=[]
    # Entire list declared before running; selection never consults attack metrics.
    candidates=[(30,120),(300,600),(600,1200),(900,1700),(1790,1799),(1800,)]
    dates={'theia':['2018-04-03','2018-04-04','2018-04-05'],
           'cadets':['2018-04-03','2018-04-04']}
    for dataset,days in dates.items():
        for date in days:
            day=EventDay(ROOT/('data/processed/'+dataset+'_e3'),date)
            ids=np.random.RandomState(17).choice(day.targets,1000,replace=False)
            spans=[]
            for i in ids:
                cp=day.clean_positions(int(i))
                spans.append((int(day.events['t'][i])-int(day.events['t'][cp[0]]))/NS if len(cp) else 0.)
            for durations in candidates:
                active=degenerate=0;keff=[];kraw=[]
                for i in ids:
                    _,_,s=draw_contexts(day,int(i),'dispersed',23,0,durations,.2)
                    active+=int(s['K_eff']>0);degenerate+=s['degenerate']
                    keff.append(s['K_eff']);kraw.append(s['K_raw'])
                row=dict(dataset=dataset,date=date,n=len(ids),durations=list(durations),
                         active=active,active_rate=active/len(ids),degenerate=degenerate,
                         nondegenerate_active=active-degenerate,
                         mean_K_eff=float(np.mean(keff)),mean_K_raw=float(np.mean(kraw)),
                         clean_tail_span_seconds_quantiles=np.quantile(spans,[0,.25,.5,.75,.95,1]).tolist())
                results.append(row);print(json.dumps(row),flush=True)
            day.close()
    out=ROOT/'refine-logs/TRAIN_MASK_DIAGNOSTIC.json'
    out.write_text(json.dumps(dict(scope='benign training only; no test labels or detector results',
                    target_seed=17,mask_seed=23,clean_probability=.2,context=128,lookback_seconds=1800,
                    candidates=candidates,results=results,seconds=time.monotonic()-started),indent=2),encoding='utf-8')
if __name__=='__main__':main()
