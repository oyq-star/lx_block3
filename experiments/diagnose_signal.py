"""Retrospective path novelty and strict-past sibling coverage on the pilot cohort."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation import process_metrics


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def columns(root, date, names):
    day = root / ('day_' + date)
    if not (day / 'INDEX_COMPLETE').exists():
        raise RuntimeError('Day not indexed: ' + date)
    return {name: np.load(day / (name + '.npy'), mmap_mode='r') for name in names}


def train_actors(root, dates):
    actors = set()
    for date in dates:
        data = columns(root, date, ('actor', 'targets'))
        indices = data['targets']
        for offset in range(0, len(indices), 200000):
            actors.update(np.unique(data['actor'][indices[offset:offset+200000]]).tolist())
    return actors


def path_novelty(root, dates, universe, positives, test_actors):
    seen_actors = train_actors(root, dates)
    requested = seen_actors | test_actors
    paths = {}
    with (root / 'subjects.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            actor = int(row['index_id'])
            path = row['path'].strip()
            if actor in requested and path.lower() not in ('', 'null', 'none', chr(92)+'n'):
                paths[actor] = path
    known = {paths[actor] for actor in seen_actors if actor in paths}
    scores = {actor: float(paths[actor] not in known) for actor in test_actors if actor in paths}
    return dict(scope='fixed dump subject paths; temporal validity not established',
                training_actors=len(seen_actors), seen_paths=len(known),
                test_actors_missing_path=len(test_actors)-len(scores),
                novel_test_actors=int(sum(scores.values())),
                novel_annotated_actors=int(sum(scores.get(actor, 0) for actor in positives)),
                metrics=process_metrics(universe, positives, scores))


def birth_records(root, dates, origin_ids):
    records = []
    for date in dates:
        data = columns(root, date, ('actor', 'object', 'rel', 'otype', 't', 'targets'))
        indices = data['targets']
        for offset in range(0, len(indices), 200000):
            batch = indices[offset:offset+200000]
            batch = batch[np.isin(data['rel'][batch], origin_ids) & (data['otype'][batch] == 1)]
            for edge in batch:
                parent, child = int(data['actor'][edge]), int(data['object'][edge])
                if parent != child:
                    records.append((int(data['t'][edge]), parent, child))
    return sorted(records)


def sibling_coverage(root, dates, test_start, test_end, indices, actors, positives, origin_ids):
    births = birth_records(root, dates, origin_ids)
    test_day = columns(root, dates[-1], ('t', 'starts', 'ends'))
    parents = {}
    for birth, parent, child in births:
        if birth >= test_end:
            break
        previous = parents.setdefault(child, (parent, birth))
        if previous[0] != parent:
            parents[child] = (-1, 0)
    eligible = {child: (parent, birth) for child, (parent, birth) in parents.items() if parent >= 0}
    first_behaviors = {}
    for child, (parent, birth) in eligible.items():
        lower, upper = test_day['starts'][child], test_day['ends'][child]
        if lower < 0:
            continue
        times = test_day['t'][lower:upper]
        first = np.searchsorted(times, max(birth, test_start-1800*10**9), side='right')
        if first < len(times):
            first_behaviors.setdefault(parent, []).append((int(times[first]), child))
    first_two = {parent: sorted(records)[:2] for parent, records in first_behaviors.items()}
    times = test_day['t'][indices]
    cuts = np.r_[0, np.flatnonzero(actors[1:] != actors[:-1])+1, len(actors)]
    counts = {'annotated': Counter(), 'unannotated': Counter()}
    for begin, end in zip(cuts[:-1], cuts[1:]):
        actor = int(actors[begin])
        group = counts['annotated' if actor in positives else 'unannotated']
        group.update(processes=1, events=int(end-begin))
        if actor not in eligible:
            continue
        parent, birth = eligible[actor]
        actor_times = times[begin:end]
        parent_events = len(actor_times)-int(np.searchsorted(actor_times, birth, side='right'))
        if parent_events:
            group.update(processes_with_parent=1, events_with_parent=parent_events)
        sibling = next((value for value, child in first_two.get(parent, ()) if child != actor), None)
        if sibling is not None:
            sibling_events = len(actor_times)-int(np.searchsorted(actor_times, max(birth, sibling), side='right'))
            if sibling_events:
                group.update(processes_with_prior_sibling=1, events_with_prior_sibling=sibling_events)
    return dict(scope='observed clone/fork candidate links; sibling activity starts no earlier than 30 minutes before test window',
                limits='offline cohort excludes any child with conflicting parents through test end; no sibling feature trained; no inference about missing links',
                indexed_birth_days=dates, creation_records_before_test_end=sum(birth < test_end for birth, _, _ in births),
                ambiguous_children=len(parents)-len(eligible),
                counts={label: {key: group[key] for key in ('processes','events','processes_with_parent','events_with_parent',
                        'processes_with_prior_sibling','events_with_prior_sibling')} for label, group in counts.items()})


def main():
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    split = read_json(args.cache / 'split.json')
    root = ROOT / ('data/processed/' + split['dataset'] + '_e3')
    cohort = read_json(args.cache / 'cohort.json')
    test = args.cache / 'test_clean'
    indices = np.load(test / 'indices.npy', mmap_mode='r')
    actors = np.load(test / 'actors.npy', mmap_mode='r')
    positives = set(cohort['positive_processes'])
    origin_ids = [int(index) for index, label in read_json(root / 'schema.json')['relation_names'].items()
                  if label in {'EVENT_CLONE', 'EVENT_FORK'}]
    if not origin_ids:
        raise RuntimeError('No clone/fork relation in the input schema')
    bounds = [split['test_start_utc'], split['test_end_utc']]
    start, end = [int(datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())*10**9 for value in bounds]
    days = split['train_dates'] + [split['validation_date'], split['test_date']]
    result = dict(dataset=split['dataset'], test_window=bounds,
                  path_novelty=path_novelty(root, split['train_dates'], cohort['universe'], positives, set(np.unique(actors).tolist())),
                  sibling_coverage=sibling_coverage(root, days, start, end, indices, actors, positives, origin_ids))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
