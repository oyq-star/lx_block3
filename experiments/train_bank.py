"""Materialize target histories once for all model initialization seeds."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from causal_data import EventDay, draw_contexts, encode_positions


def warm_bank(cache, mode, epochs=3):
    root=Path(__file__).resolve().parents[1]
    split=json.loads((cache/'split.json').read_text())
    output=cache/'train_bank_v2'/mode
    output.mkdir(parents=True,exist_ok=True)
    mapping=np.load(cache/'token_map.npy')
    rows=np.load(cache/'train_targets.npy')
    source=root/('data/processed/'+split['dataset']+'_e3')
    days=[EventDay(source,date,split['context'],split['lookback_seconds']) for date in split['train_dates']]
    try:
        for epoch in range(epochs):
            destination=output/(str(epoch)+'.npz')
            if destination.exists():continue
            started=time.monotonic()
            clean=np.zeros((len(rows),split['context']),dtype=np.uint16)
            aug=np.zeros_like(clean);lengths=np.zeros(len(rows),dtype=np.uint16)
            aug_lengths=np.zeros_like(lengths);labels=np.zeros_like(lengths)
            totals=dict(K_raw=0,K_eff=0,empty=0,degenerate=0,targets=len(rows))
            for row,(date,index) in enumerate(rows):
                day=days[int(date)]
                original,modified,summary=draw_contexts(day,int(index),mode,split['mask_seed'],epoch,(1790,1799),.2)
                first=encode_positions(day,original,mapping);second=encode_positions(day,modified,mapping)
                clean[row,:len(first)]=first;aug[row,:len(second)]=second
                lengths[row]=len(first);aug_lengths[row]=len(second)
                labels[row]=mapping[int(day.events['rel'][index])*4+int(day.events['otype'][index])]
                for key in ('K_raw','K_eff','empty','degenerate'):totals[key]+=summary[key]
            temporary=destination.with_suffix('.tmp.npz')
            np.savez(temporary,clean=clean,aug=aug,lengths=lengths,aug_lengths=aug_lengths,labels=labels,**totals)
            temporary.replace(destination)
            print(json.dumps(dict(mode=mode,epoch=epoch,seconds=time.monotonic()-started,**totals)),flush=True)
    finally:
        for day in days:day.close()
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--modes',default='clean,iid,dispersed,crop,block')
    args=parser.parse_args()
    for mode in args.modes.split(','):warm_bank(args.cache,mode)
