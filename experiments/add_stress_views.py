"""Build held-out-duration and stream-consistent evaluation views."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import psutil

from build_caches import mmap_array, write_eval
from causal_data import EventDay, NS


ROOT = Path(__file__).resolve().parents[1]


def utc_ns(value):
    return int(dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()) * NS


def stream_cache(day, targets, mapping, folder, config):
    if (folder/'CACHE_COMPLETE').exists():
        return
    start = time.monotonic()
    times = day.events['t']
    left, right = config['interval_ns']
    removed = (times >= left) & (times < right)
    if config['relation_id'] is not None:
        removed &= day.events['rel'] == config['relation_id']
    actors_original = day.events['actor'][targets]
    kept = targets[~removed[targets]]
    actors = day.events['actor'][kept]
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'indices.npy', kept)
    np.save(folder/'actors.npy', actors)
    np.save(folder/'seq.npy', day.events['seq'][kept])
    np.save(folder/'targets.npy', mapping[day.raw_tokens(kept)])
    contexts = mmap_array(folder/'contexts.npy', np.uint16, (len(kept), day.k))
    lengths = mmap_array(folder/'lengths.npy', np.uint16, (len(kept),))
    width = np.arange(day.k)
    refill = changed = empty = 0
    cuts = np.r_[0, np.flatnonzero(actors[1:] != actors[:-1])+1, len(actors)] if len(actors) else [0]
    for first, last in zip(cuts[:-1], cuts[1:]):
        actor = actors[first]
        lo, hi = day.starts[actor], day.ends[actor]
        positions = np.flatnonzero(~removed[lo:hi])+lo
        retained_times = times[positions]
        original_times = times[lo:hi]
        encoded = mapping[day.raw_tokens(positions)]
        for a in range(first, last, 4096):
            b = min(last, a+4096)
            target_times = times[kept[a:b]]
            before = np.searchsorted(retained_times, target_times, side='left')
            after = np.searchsorted(retained_times, target_times-day.lookback, side='left')
            length = np.minimum(day.k, before-after)
            index = before[:, None]-length[:, None]+width
            valid = width < length[:, None]
            safe = np.minimum(index, len(encoded)-1)
            matrix = np.where(valid, encoded[safe], 0)
            matrix[length == 0, 0] = 1
            contexts[a:b] = matrix
            lengths[a:b] = np.maximum(1, length)
            clean_end = np.searchsorted(original_times, target_times, side='left') + lo
            clean_start = np.maximum(clean_end-day.k, np.searchsorted(original_times, target_times-day.lookback)+lo)
            replaced = (valid & (positions[safe] < clean_start[:, None])).sum(axis=1)
            refill += int(replaced.sum())
            changed += int(((length != clean_end-clean_start) | (replaced > 0)).sum())
            empty += int((length == 0).sum())
    contexts.flush()
    lengths.flush()
    original_ids, original_counts = np.unique(actors_original, return_counts=True)
    visible_ids, visible_counts = np.unique(actors, return_counts=True)
    np.savez(folder/'process_event_counts.npz', original_ids=original_ids, original_counts=original_counts,
             visible_ids=visible_ids, visible_counts=visible_counts)
    info = dict(**config, regime='single shared mask applied before context and target construction',
                rows=len(kept), original_targets=len(targets), deleted_targets=len(targets)-len(kept),
                original_actor_count=len(original_ids), visible_actor_count=len(visible_ids),
                refill_tokens=refill, changed_contexts=changed, empty_contexts=empty,
                elapsed_seconds=time.monotonic()-start, rss_MiB=psutil.Process().memory_info().rss/2**20)
    (folder/'metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (folder/'CACHE_COMPLETE').write_text('complete\n')
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--views',default='global_relation,global_time,unseen300')
    args = parser.parse_args()
    cache = args.cache
    split = json.loads((cache/'split.json').read_text())
    root = ROOT/('data/processed/'+split['dataset']+'_e3')
    schema = json.loads((root/'schema.json').read_text())
    rels = {int(key):value for key,value in schema['relation_names'].items()}
    start,end = map(utc_ns,[split['test_start_utc'],split['test_end_utc']])
    day = EventDay(root,split['test_date'],split['context'],split['lookback_seconds'])
    targets = np.load(cache/'test_clean/indices.npy',mmap_mode='r')
    mapping = np.load(cache/'token_map.npy')
    counts = np.zeros(max(rels)+1,dtype=np.int64)
    for date in split['train_dates']:
        rel = np.load(root/('day_'+date)/'rel.npy',mmap_mode='r')
        for left in range(0,len(rel),200000):
            counts += np.bincount(rel[left:left+200000],minlength=len(counts))
    relation = int(np.argmax(counts))
    for name in args.views.split(','):
        folder=cache/('test_'+name)
        if name=='unseen300':
            write_eval(day,targets,mapping,folder,'block',split['mask_seed'],300)
            continue
        if name not in {'global_relation','global_time'}:
            raise ValueError(name)
        duration=split['duration_seconds'] if name=='global_relation' else 300
        offset=int(np.random.RandomState(split['mask_seed']).randint(0,(end-start)//NS-duration+1))
        left=start+offset*NS
        stream_cache(day,targets,mapping,folder,dict(mode=name,mask_seed=split['mask_seed'],
            relation_id=relation if name=='global_relation' else None,
            relation_name=rels[relation] if name=='global_relation' else None,
            relation_choice='most frequent in indexed training-day records',
            duration_seconds=duration,interval_ns=[left,left+duration*NS]))
    day.close()


if __name__=='__main__':main()
