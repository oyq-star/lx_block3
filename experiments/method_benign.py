"""Create disjoint benign scoring windows and cache frozen GRU predictions."""

import argparse
from contextlib import closing

import numpy as np
import torch

from add_stress_views import stream_cache, utc_ns
from causal_data import EventDay, positive_processes
from outage_replay import frozen_grus
from semantic_fusion import ROOT, RUN, load, save_json
from train_eval import cache_arrays, score_cache


OUTPUT = ROOT / 'runs/method_followup_v1'
WINDOWS = {
    'theia': dict(reference=('12:30','14:30'), threshold=('15:00','17:00'), audit=('20:30','22:30')),
    'cadets': dict(reference=('09:00','11:00'), threshold=('12:00','14:00'), audit=('18:00','20:00')),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=tuple(WINDOWS),required=True)
    args = parser.parse_args()
    dataset = args.dataset
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    cache = ROOT/'data/cache'/f'{dataset}_longoutage_v1'
    source = ROOT/'data/processed'/f'{dataset}_e3'
    split = load(cache/'split.json')
    date = split['validation_date']
    if positive_processes(source,date):
        raise ValueError(f'Annotated positives on selected benign day {date}')
    mapping = np.load(cache/'token_map.npy')
    models = frozen_grus(RUN/dataset,device)
    with closing(EventDay(source,date,split['context'],split['lookback_seconds'])) as day:
        for name, hours in WINDOWS[dataset].items():
            folder = OUTPUT/dataset/'benign'/name
            result = folder/'scores_complete.json'
            if result.is_file():
                continue
            start, end = (utc_ns(date+'T'+h+':00Z') for h in hours)
            targets = day.target_range(start,end)
            config = dict(mode='clean_benign',interval_ns=[end,end],relation_id=None)
            stream_cache(day,targets,mapping,folder,config)
            np.testing.assert_array_equal(np.load(folder/'indices.npy',mmap_mode='r'),targets)
            arrays = cache_arrays(folder)
            for seed, model in models.items():
                _, profile = score_cache(model,arrays,device,1024,folder/f'{seed}_nll.npy')
                save_json(folder/f'{seed}_profile.json',profile)
            save_json(result,dict(date=date,window=hours,original_targets=len(targets),
                                 actors=len(np.unique(arrays['actors'])),device=torch.cuda.get_device_name(0)))
            del arrays
            print(dataset,name,hours,len(targets),flush=True)


if __name__ == '__main__':
    main()
