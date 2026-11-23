"""Score process windows using benign provenance profiles and frozen GRU evidence."""

import argparse
from collections import defaultdict
import csv
from hashlib import blake2b
import json
from pathlib import Path
import posixpath
import time

import joblib
import numpy as np
import psutil

from count_baseline import score as count_scores
from evaluation import process_metrics
from heldout_campaigns import RUN, WINDOWS, load
from prepare_dataset import copy_rows


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 23, 42)
VIEWS = ('clean', 'iid', 'dispersed', 'block', 'unseen300', 'global_relation', 'global_time')
MIX = np.uint64(11400714819323198485)
UNKNOWN_PATHS = {'', 'null', 'none', '<unknown>', 'unknown', r'\n'}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def key(value):
    if not value:
        return 0
    return int.from_bytes(blake2b(value.encode('utf-8'), digest_size=8).digest(), 'little')


def path_fields(path):
    candidate = path.strip()
    if candidate.lower() in UNKNOWN_PATHS:
        return 0, 0, 0
    normalized = posixpath.normpath(candidate)
    return key(normalized), key(posixpath.basename(normalized)), key(posixpath.dirname(normalized))


def path_index(dataset, processed, output):
    schema = load(processed / 'schema.json')
    size = schema['max_node_index'] + 1
    subjects = np.zeros((size, 3), dtype=np.uint64)
    with (processed / 'subjects.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            subjects[int(row['index_id'])] = path_fields(row['path'])
    filename = output / 'file_keys.npy'
    if not filename.exists():
        keys = np.zeros((size, 3), dtype=np.uint64)
        for fields in copy_rows(ROOT / 'data/downloads' / (dataset + '_e3.dump'), 'file_node_table', output):
            if len(fields) != 4:
                raise ValueError('Unexpected file-node table schema')
            keys[int(fields[-1])] = path_fields(fields[2].decode('utf-8', 'replace'))
        temporary = filename.with_suffix('.tmp.npy')
        np.save(temporary, keys)
        temporary.replace(filename)
        del keys
    return subjects, np.load(filename, mmap_mode='r')


def columns(processed, date, names):
    folder = processed / ('day_' + date)
    if not (folder / 'INDEX_COMPLETE').exists():
        raise FileNotFoundError(folder / 'INDEX_COMPLETE')
    return [np.load(folder / (name + '.npy'), mmap_mode='r') for name in names]


def incidence(processed, date, indices, files, spawn_relations):
    actors, objects, relations, object_types = columns(processed, date, ('actor', 'object', 'rel', 'otype'))
    file_edges = []
    spawn_edges = []
    for start in range(0, len(indices), 200_000):
        rows = indices[start:start + 200_000]
        actor, obj, relation, kind = actors[rows], objects[rows], relations[rows], object_types[rows]
        is_file = (kind == 2) & (files[obj, 0] != 0)
        is_spawn = (kind == 1) & np.isin(relation, spawn_relations) & (actor != obj)
        if is_file.any():
            file_edges.append(np.unique(np.column_stack((actor[is_file], relation[is_file], obj[is_file])), axis=0))
        if is_spawn.any():
            spawn_edges.append(np.unique(np.column_stack((actor[is_spawn], obj[is_spawn])), axis=0))
    unique = lambda parts, width: np.unique(np.vstack(parts), axis=0) if parts else np.empty((0, width), dtype=np.int32)
    return np.unique(actors[indices]), unique(file_edges, 3), unique(spawn_edges, 2)


def lift(values, group):
    return (values * MIX) ^ (group.astype(np.uint64) + np.uint64(1))


def file_signatures(edges, subjects, files):
    actor, relation, obj = edges.T
    return (
        lift(files[obj, 0], relation),
        lift(files[obj, 1], relation),
        lift(files[obj, 2], relation),
        lift(files[obj, 0], relation) ^ (subjects[actor, 0] * np.uint64(6364136223846793005)),
    )


def spawn_signature(edges, subjects):
    parent, child = edges.T
    valid = (subjects[parent, 0] != 0) & (subjects[child, 0] != 0)
    return child[valid], (subjects[parent[valid], 0] * MIX) ^ subjects[child[valid], 0]


def document_support(actors, signatures):
    documents = np.empty(len(actors), dtype=[('actor', np.int32), ('signature', np.uint64)])
    documents['actor'] = actors
    documents['signature'] = signatures
    unique = np.unique(documents)
    keys, counts = np.unique(unique['signature'], return_counts=True)
    return keys, counts


def merge_support(entries):
    keys = np.concatenate([item[0] for item in entries])
    frequency = np.concatenate([item[1] for item in entries])
    if not len(keys):
        return keys, frequency
    order = np.argsort(keys)
    ordered = keys[order]
    cuts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return ordered[cuts], np.add.reduceat(frequency[order], cuts)


def surprise(reference, values, documents):
    keys, counts = reference
    if not len(keys):
        return np.full(len(values), np.log1p(documents))
    index = np.minimum(np.searchsorted(keys, values), len(keys) - 1)
    support = np.where(keys[index] == values, counts[index], 0)
    return np.log((documents + 1) / (1 + support))


def fit_semantics(processed, dates, subjects, files, spawn_relations):
    supports = defaultdict(list)
    documents = 0
    for date in dates:
        indices, = columns(processed, date, ('targets',))
        actors, edges, spawns = incidence(processed, date, indices, files, spawn_relations)
        documents += len(actors)
        known = subjects[actors, 0] != 0
        for dimension in range(3):
            supports['image' + str(dimension)].append(document_support(actors[known], subjects[actors[known], dimension]))
        for dimension, values in enumerate(file_signatures(edges, subjects, files)):
            supports['file' + str(dimension)].append(document_support(edges[:, 0], values))
        children, signatures = spawn_signature(spawns, subjects)
        supports['spawn'].append(document_support(children, signatures))
        print(json.dumps(dict(stage='fit_semantics', date=date, actors=len(actors), unique_file_edges=len(edges))), flush=True)
    return {name: merge_support(entries) for name, entries in supports.items()}, documents


def record_peak(actor_ids, values, target_ids):
    scores = np.zeros(len(target_ids))
    if not len(actor_ids) or not len(target_ids):
        return scores, np.zeros(len(target_ids), dtype=bool)
    positions = np.searchsorted(target_ids, actor_ids)
    safe = np.minimum(positions, len(target_ids) - 1)
    included = (positions < len(target_ids)) & (target_ids[safe] == actor_ids)
    np.maximum.at(scores, positions[included], values[included])
    present = np.zeros(len(target_ids), dtype=bool)
    present[positions[included]] = True
    return scores, present


def semantic_evidence(processed, date, indices, subjects, files, spawn_relations, supports, documents):
    actors, edges, spawns = incidence(processed, date, indices, files, spawn_relations)
    known = subjects[actors, 0] != 0
    image_surprise = sum(weight * surprise(supports['image' + str(dimension)], subjects[actors, dimension], documents)
                         for dimension, weight in enumerate((.5, .25, .25)))
    signature_surprise = sum(weight * surprise(supports['file' + str(dimension)], values, documents)
                             for dimension, (weight, values) in enumerate(zip((.35, .15, .2, .3), file_signatures(edges, subjects, files))))
    children, signatures = spawn_signature(spawns, subjects)
    return (actors, np.where(known, image_surprise, np.nan),
            (edges[:, 0], signature_surprise), (children, surprise(supports['spawn'], signatures, documents)))


def semantic_scores(processed, date, indices, subjects, files, spawn_relations, supports, documents):
    actors, image, files_evidence, creation = semantic_evidence(
        processed, date, indices, subjects, files, spawn_relations, supports, documents
    )
    columns = [image]
    for associated_actors, values in (files_evidence, creation):
        peaks, present = record_peak(associated_actors, values, actors)
        columns.append(np.where(present, peaks, np.nan))
    return actors, np.column_stack(columns)


def tail(reference, values):
    """Map higher anomalies to smaller normal tail probabilities with midranks for ties."""
    normal = np.sort(reference[np.isfinite(reference)])
    available = np.isfinite(values)
    result = np.zeros(len(values))
    if not len(normal):
        return result
    observed = values[available]
    rank = (np.searchsorted(normal, observed, side='left') + np.searchsorted(normal, observed, side='right')) / 2
    probability = (len(normal) - rank + .5) / (len(normal) + 1)
    spread = max(normal[-1] - np.quantile(normal, .9), np.std(normal), 1e-6)
    excess = np.log1p(np.maximum(observed - normal[-1], 0) / spread)
    result[available] = -np.log(probability) + np.minimum(excess, 2)
    return result


def semantic_tail(reference, values):
    return np.column_stack([tail(reference[:, dimension], values[:, dimension]) for dimension in range(values.shape[1])]).max(axis=1)


def align(actors, scores):
    return np.array([scores.get(int(actor), np.nan) for actor in actors])


def process_nll(folder, path, actors):
    ids = np.load(folder / 'actors.npy', mmap_mode='r')
    loss = np.load(path, mmap_mode='r')
    if len(ids) != len(loss):
        raise ValueError('GRU score cache and target index differ')
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    if not np.array_equal(ids[cuts], actors):
        raise ValueError('GRU and semantic actor indexes differ')
    return np.maximum.reduceat(loss, cuts)


def combine(semantic, counts, sequence):
    return {
        'fusion': np.maximum(semantic, counts) + .25 * np.minimum(semantic, counts) + .125 * sequence,
        'semantic_only': semantic,
        'count_plus_gru': counts + .125 * sequence,
        'semantic_plus_count': np.maximum(semantic, counts) + .25 * np.minimum(semantic, counts),
    }


def evaluate(dataset, output, calibration_only=False):
    start = time.perf_counter()
    cache = ROOT / 'data/cache' / (dataset + '_longoutage_v1')
    split = load(cache / 'split.json')
    primary = RUN / dataset
    processed = ROOT / 'data/processed' / (dataset + '_e3')
    schema = load(processed / 'schema.json')
    relation_ids = [int(index) for index, name in schema['relation_names'].items()
                    if name in {'EVENT_CLONE', 'EVENT_FORK'}]
    output.mkdir(parents=True, exist_ok=True)
    subjects, files = path_index(dataset, processed, output)
    references, documents = fit_semantics(processed, split['train_dates'], subjects, files, relation_ids)
    forest = joblib.load(primary / 'isolation_forest/model.joblib')
    vocabulary = load(primary / 'isolation_forest/results.json')['vocabulary']

    def inputs(date, folder):
        indices = np.load(folder / 'indices.npy', mmap_mode='r')
        ids, semantic = semantic_scores(processed, date, indices, subjects, files, relation_ids, references, documents)
        return ids, semantic, align(ids, count_scores(folder, forest, vocabulary))

    val_folder = cache / 'validation'
    val_actors, val_semantic, val_counts = inputs(split['validation_date'], val_folder)
    val_semantic_score = semantic_tail(val_semantic, val_semantic)
    val_count_score = tail(val_counts, val_counts)
    val_sequence = {
        seed: process_nll(val_folder, primary / f'{seed}_block/validation_nll.npy', val_actors)
        for seed in SEEDS
    }
    np.savez(output / 'validation_scores.npz', actors=val_actors, semantic=val_semantic_score, count=val_count_score,
             **{f'temporal_{seed}': tail(val_sequence[seed], val_sequence[seed]) for seed in SEEDS})
    if calibration_only:
        return
    thresholds = {seed: {name: float(np.quantile(scores, .99))
                        for name, scores in combine(val_semantic_score, val_count_score,
                                                     tail(val_sequence[seed], val_sequence[seed])).items()}
                  for seed in SEEDS}
    save_json(output / 'calibration.json', dict(validation_processes=len(val_actors), training_process_days=documents,
              validation_date=split['validation_date'], training_dates=split['train_dates'], quantile=.99,
              thresholds=thresholds, fixed_coefficients=[1, .25, .125],
              input_scope='Retrospective observed process and event endpoints, with dump path metadata.'))
    scenarios = []
    for view in VIEWS:
        scenarios.append((split['test_date'], view, cache / ('test_' + view), cache / 'cohort.json', primary))
    for name, date, _, _ in WINDOWS:
        if name == dataset:
            heldout = ROOT / 'data/cache/heldout' / dataset / date
            scenarios.append((date, 'heldout_clean', heldout, heldout / 'cohort.json', RUN / 'heldout' / dataset / date))

    rows = []
    target_cache = {}
    clean_indices = np.load(cache / 'test_clean/indices.npy', mmap_mode='r')
    for date, view, folder, cohort_file, scores_source in scenarios:
        signature = 'main_fixed_targets' if view in {'clean', 'iid', 'dispersed', 'block', 'unseen300'} else (date, view)
        if signature == 'main_fixed_targets' and not np.array_equal(clean_indices, np.load(folder / 'indices.npy', mmap_mode='r')):
            raise ValueError('Fixed-target view differs from the clean target list')
        if signature not in target_cache:
            target_cache[signature] = inputs(date, folder)
        actors, semantics, counts = target_cache[signature]
        observed_actors = np.unique(np.load(folder / 'actors.npy', mmap_mode='r'))
        if not np.array_equal(actors, observed_actors):
            raise ValueError('View actor IDs do not match dump targets')
        semantic = semantic_tail(val_semantic, semantics)
        count = tail(val_counts, counts)
        cohort = load(cohort_file)
        scenario = output / (date + '_' + view)
        scenario.mkdir(exist_ok=True)
        for seed in SEEDS:
            path = scores_source / (f'{seed}_block_event_nll.npy' if view == 'heldout_clean' else f'{seed}_block/{view}_event_nll.npy')
            sequence = tail(val_sequence[seed], process_nll(folder, path, actors))
            candidates = combine(semantic, count, sequence)
            np.savez(scenario / f'{seed}_scores.npz', actors=actors, count=count, semantic=semantic,
                     temporal=sequence, raw_semantic=semantics, **candidates)
            for method, values in candidates.items():
                scores = dict(zip(actors.tolist(), values.tolist()))
                metrics = process_metrics(cohort['universe'], cohort['positive_processes'], scores,
                                          threshold=thresholds[seed][method])
                rows.append(dict(dataset=dataset, date=date, view=view, seed=seed, method=method, metrics=metrics))
        result = [row['metrics']['AP_all'] for row in rows[-len(SEEDS) * 4:] if row['method'] == 'fusion']
        print(json.dumps(dict(stage='evaluate', dataset=dataset, date=date, view=view, fusion_AP=result)), flush=True)
    save_json(output / 'results.json', dict(status='COMPLETED', dataset=dataset, rows=rows,
              budget=dict(wall_seconds=time.perf_counter() - start, rss_end_MiB=psutil.Process().memory_info().rss / 2**20,
                          device='cpu, reused frozen GRU predictions')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('theia', 'cadets'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/semantic_fusion_v1')
    parser.add_argument('--calibration-only', action='store_true')
    args = parser.parse_args()
    evaluate(args.dataset, args.output / args.dataset, args.calibration_only)


if __name__ == '__main__':
    main()
