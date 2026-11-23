"""Isolation Forest on per-process event counts, with chronological benign fitting."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from add_stress_views import utc_ns
from causal_data import EventDay
from evaluation import process_metrics
from local_grid import VIEWS


ROOT=Path(__file__).resolve().parents[1]


def features(actors,targets,vocabulary):
    ids,rows=np.unique(actors,return_inverse=True)
    counts=np.bincount(rows*vocabulary+targets,minlength=len(ids)*vocabulary).reshape(len(ids),vocabulary)
    totals=counts.sum(1,keepdims=True)
    return ids,np.hstack((np.log1p(counts),counts/np.maximum(totals,1),np.log1p(totals)))


def score(folder,model,vocabulary):
    actors=np.load(folder/'actors.npy',mmap_mode='r')
    targets=np.load(folder/'targets.npy',mmap_mode='r')
    ids,matrix=features(actors,targets,vocabulary)
    return {int(actor):float(value) for actor,value in zip(ids,-model.score_samples(matrix))} if len(ids) else {}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    start=time.monotonic()
    cache=args.cache;out=args.output;out.mkdir(parents=True,exist_ok=True)
    def load(path):return json.loads(path.read_text(encoding='utf-8'))
    dataset=load(cache/'dataset.json')
    split=dataset['split'];vocab=dataset['vocabulary_size']
    training=[];day_rows=[]
    source=ROOT/('data/processed/'+split['dataset']+'_e3')
    mapping=np.load(cache/'token_map.npy')
    for date in split['train_dates']:
        day=EventDay(source,date)
        begin,end=(utc_ns(date+split[key][10:]) for key in ('test_start_utc','test_end_utc'))
        positions=day.target_range(begin,end)
        if len(positions):
            ids,matrix=features(day.events['actor'][positions],mapping[day.raw_tokens(positions)],vocab)
            training.append(matrix)
            day_rows.append(dict(date=date,events=len(positions),actor_processes=len(ids)))
        day.close()
    model=IsolationForest(n_estimators=100,max_samples=256,random_state=17,n_jobs=2).fit(np.vstack(training))
    validation=score(cache/'validation',model,vocab)
    alpha=.001 if len(validation)>=10000 else .01
    threshold=float(np.quantile(list(validation.values()),1-alpha))
    cohort=load(cache/'cohort.json');views={}
    for label in VIEWS.split(','):
        scores=score(cache/('test_'+label),model,vocab)
        views[label]=process_metrics(cohort['universe'],cohort['positive_processes'],scores,threshold=threshold)
        (out/(label+'_process_scores.json')).write_text(json.dumps(scores),encoding='utf-8')
    joblib.dump(model,out/'model.joblib')
    result=dict(status='COMPLETED',family='count baseline',seed=17,vocabulary=vocab,fit_windows=day_rows,
                score='negative forest score; log1p relation/object counts, proportions, and total event count',
                scope='end-window process count summary; not causal event scoring',
                threshold=threshold,views=views,total_seconds=time.monotonic()-start)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(dataset=split['dataset'],train_windows=day_rows,AP={k:v['AP_all'] for k,v in views.items()})),flush=True)


if __name__=='__main__':main()
