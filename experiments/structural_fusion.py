"""Add train-calibrated unique-edge and neighborhood statistics to semantic fusion."""

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
from semantic_fusion import ROOT, SEEDS, VIEWS, WINDOWS, load, save_json, tail


METHOD = 'structural_fusion'
ABLATION = 'structural_fusion_without_structure'


def features(day, indices, mapping, vocabulary, degree_scope="window"):
    actor = day.events['actor'][indices]
    nodes = day.events['object'][indices]
    targets = mapping[day.raw_tokens(indices)]
    ids, inverse = np.unique(actor, return_inverse=True)
    counts = np.bincount(inverse * vocabulary + targets, minlength=len(ids) * vocabulary).reshape(len(ids), vocabulary)
    edges = np.empty(len(indices), dtype=[('actor', np.int32), ('token', np.uint16), ('object', np.int32)])
    edges['actor'] = inverse
    edges['token'] = targets
    edges['object'] = nodes
    unique = np.unique(edges)
    edge_rows = unique['actor'].astype(np.int64)
    distinct = np.bincount(edge_rows * vocabulary + unique['token'],
                           minlength=len(ids) * vocabulary).reshape(len(ids), vocabulary)
    total_events = counts.sum(axis=1)
    total_distinct = distinct.sum(axis=1)
    if degree_scope == 'window':
        degree_key = unique['object']
    elif degree_scope == 'actor':
        degree_key = unique[['actor', 'object']]
    else:
        raise ValueError(f'Unknown object degree scope: {degree_scope}')
    _, object_rows, object_degree = np.unique(degree_key, return_inverse=True, return_counts=True)
    degree = np.log1p(object_degree[object_rows])
    neighbor_mean = np.bincount(edge_rows, weights=degree, minlength=len(ids)) / np.maximum(total_distinct, 1)
    private_fraction = np.bincount(edge_rows, weights=(object_degree[object_rows] == 1),
                                  minlength=len(ids)) / np.maximum(total_distinct, 1)
    times = day.events['t'][indices]
    starts = np.r_[0, np.flatnonzero(actor[1:] != actor[:-1]) + 1]
    if not np.array_equal(actor[starts], ids):
        raise ValueError('Expected actor-sorted targets')
    ends = np.r_[starts[1:] - 1, len(times) - 1]
    span = (times[ends] - times[starts]) / 1e9
    structural_features = np.column_stack((np.log1p(distinct), distinct / np.maximum(counts, 1),
        np.log1p(total_distinct), np.log1p(total_events), np.log1p(span), neighbor_mean, private_fraction))
    return ids, structural_features


def evaluate(dataset, output, v1):
    start = time.perf_counter()
    folder = output / dataset
    folder.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'data/cache' / (dataset + '_longoutage_v1')
    processed = ROOT / 'data/processed' / (dataset + '_e3')
    split = load(cache / 'split.json')
    vocabulary = load(cache / 'dataset.json')['vocabulary_size']
    mapping = np.load(cache / 'token_map.npy')
    reference = []
    for date in split['train_dates']:
        day = EventDay(processed, date)
        first, last = (utc_ns(date + split[edge][10:]) for edge in ('test_start_utc', 'test_end_utc'))
        indices = day.target_range(first, last)
        if len(indices):
            _, matrix = features(day, indices, mapping, vocabulary)
            reference.append(matrix)
        day.close()
    forest = IsolationForest(n_estimators=100, max_samples=256, random_state=17, n_jobs=2).fit(np.vstack(reference))
    joblib.dump(forest, folder / 'structure_forest.joblib')

    def evidence(date, data):
        day = EventDay(processed, date)
        ids, matrix = features(day, np.load(data / 'indices.npy', mmap_mode='r'), mapping, vocabulary)
        day.close()
        return ids, -forest.score_samples(matrix)

    val_ids, val = evidence(split['validation_date'], cache / 'validation')
    archive = v1 / dataset / 'validation_scores.npz'
    with np.load(archive) as val_v1:
        if not np.array_equal(val_ids, val_v1['actors']):
            raise ValueError('Structure and semantic validation actor IDs differ')
        thresholds = {}
        for seed in SEEDS:
            components = integrate(val_v1['semantic'], val_v1['count'], tail(val, val), val_v1[f'temporal_{seed}'])
            thresholds[seed] = {name: float(np.quantile(values, .99)) for name, values in components.items()}
    save_json(folder / 'calibration.json', dict(thresholds=thresholds, validation_processes=len(val),
              weights=[1, .125, .0625], seed=17, previous_run=str(v1)))
    scenarios = [(split['test_date'], view, cache / ('test_' + view), cache / 'cohort.json') for view in VIEWS]
    for name, date, _, _ in WINDOWS:
        if name == dataset:
            heldout = ROOT / 'data/cache/heldout' / dataset / date
            scenarios.append((date, 'heldout_clean', heldout, heldout / 'cohort.json'))
    rows = []
    evidence_cache = {}
    for date, view, data, cohort_file in scenarios:
        condition = (date, 'fixed_targets') if view in {'clean', 'iid', 'dispersed', 'block', 'unseen300'} else (date, view)
        if condition not in evidence_cache:
            evidence_cache[condition] = evidence(date, data)
        ids, scores = evidence_cache[condition]
        structure = tail(val, scores)
        cohort = load(cohort_file)
        scenario = folder / (date + '_' + view)
        scenario.mkdir(exist_ok=True)
        for seed in SEEDS:
            with np.load(v1 / dataset / (date + '_' + view) / f'{seed}_scores.npz') as previous:
                if not np.array_equal(ids, previous['actors']):
                    raise ValueError('Structure and semantic actor IDs differ')
                components = integrate(previous['semantic'], previous['count'], structure, previous['temporal'])
            np.savez(scenario / f'{seed}_scores.npz', actors=ids, structure=structure, **components)
            for name, combined in components.items():
                metrics = process_metrics(cohort['universe'], cohort['positive_processes'],
                                          dict(zip(ids.tolist(), combined.tolist())), threshold=thresholds[seed][name])
                rows.append(dict(dataset=dataset, date=date, view=view, method=name, seed=seed, metrics=metrics))
        current = [entry['metrics']['AP_all'] for entry in rows[-2 * len(SEEDS):] if entry['method'] == METHOD]
        print(json.dumps(dict(dataset=dataset, date=date, view=view, AP=current)), flush=True)
    save_json(folder / 'results.json', dict(status='COMPLETED', rows=rows, elapsed_seconds=time.perf_counter() - start))


def integrate(semantic, counts, structure, sequence):
    values = np.vstack((semantic, counts, structure))
    common = .0625 * sequence
    return {
        METHOD: values.max(axis=0) + .125 * np.median(values, axis=0) + common,
        ABLATION: np.maximum(semantic, counts) + .125 * np.minimum(semantic, counts) + common,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('theia', 'cadets'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/semantic_fusion_v2')
    parser.add_argument('--v1', type=Path, default=ROOT / 'runs/semantic_fusion_v1')
    args = parser.parse_args()
    evaluate(args.dataset, args.output, args.v1)


if __name__ == '__main__':
    main()
