"""Evaluate frozen detectors on additional simulated audit outages in the E3 caches."""

import argparse
from contextlib import closing
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from add_stress_views import stream_cache, utc_ns
from causal_data import EventDay, NS
from count_baseline import score as count_scores
from evaluation import process_metrics
from semantic_fusion import (
    ROOT, RUN, SEEDS, align, fit_semantics, load, path_index, save_json,
    semantic_scores, semantic_tail, tail,
)
from structural_fusion import METHOD, features, integrate
from train_eval import EventGRU, cache_arrays, score_cache


DURATIONS = (300, 900)
START_FRACTIONS = (0.2, 0.5, 0.8)
OUTPUT = ROOT / 'runs' / 'outage_replay_v1'
ADDITIONAL_OUTPUT = ROOT / 'runs' / 'outage_additional_v1'


def frozen_grus(primary, device):
    models = {}
    for seed in SEEDS:
        checkpoint = torch.load(primary / f'{seed}_block/best.pt', map_location=device)
        config = checkpoint['config']
        model = EventGRU(config['dataset']['vocabulary_size'], config['embedding'], config['hidden'])
        model.load_state_dict(checkpoint['model'])
        models[seed] = model.to(device).eval()
    return models


def window_rules(paths, counts, structure):
    evidence = np.vstack((paths, counts, structure))
    maximum = evidence.max(axis=0)
    return {
        'WindowMax': maximum,
        'WithoutGRU': maximum + .125 * np.median(evidence, axis=0),
        'WindowMean': evidence.mean(axis=0),
    }


def run(dataset, output, device, additional_date=None):
    cache = ROOT / 'data/cache' / (dataset + '_longoutage_v1')
    source = ROOT / 'data/processed' / (dataset + '_e3')
    v1 = ROOT / 'runs/semantic_fusion_v1' / dataset
    v2 = ROOT / 'runs/semantic_fusion_v2' / dataset
    primary = RUN / dataset
    split = load(cache / 'split.json')
    target = output / dataset
    if additional_date:
        date = additional_date
        additional_cache = ROOT / 'data/cache/heldout' / dataset / date
        universe = load(additional_cache / 'cohort.json')
        window = universe['window']
        targets = np.load(additional_cache / 'indices.npy', mmap_mode='r')
        target /= date
    else:
        date = split['test_date']
        universe = load(cache / 'cohort.json')
        window = [split['test_start_utc'], split['test_end_utc']]
        targets = np.load(cache / 'test_clean/indices.npy', mmap_mode='r')
    test_start, test_end = map(utc_ns, window)
    target.mkdir(parents=True, exist_ok=True)
    mapping = np.load(cache / 'token_map.npy')
    vocabulary = load(cache / 'dataset.json')['vocabulary_size']
    count_model = joblib.load(primary / 'isolation_forest/model.joblib')
    structure_model = joblib.load(v2 / 'structure_forest.joblib')
    spawn = [int(i) for i, name in load(source / 'schema.json')['relation_names'].items()
             if name in {'EVENT_CLONE', 'EVENT_FORK'}]
    subjects, files = path_index(dataset, source, v1)
    supports, documents = fit_semantics(source, split['train_dates'], subjects, files, spawn)
    models = frozen_grus(primary, device)
    thresholds = load(v2 / 'calibration.json')['thresholds']
    stored_val = np.load(v1 / 'validation_scores.npz')
    validation_folder = cache / 'validation'
    val_indices = np.load(validation_folder / 'indices.npy', mmap_mode='r')
    val_ids, val_paths = semantic_scores(
        source, split['validation_date'], val_indices, subjects, files, spawn, supports, documents
    )
    val_counts = align(val_ids, count_scores(validation_folder, count_model, vocabulary))
    with closing(EventDay(source, split['validation_date'])) as validation_day:
        str_ids, matrix = features(validation_day, val_indices, mapping, vocabulary)
        val_structure = -structure_model.score_samples(matrix)
    np.testing.assert_array_equal(val_ids, str_ids)
    np.testing.assert_array_equal(val_ids, stored_val['actors'])
    np.testing.assert_allclose(semantic_tail(val_paths, val_paths), stored_val['semantic'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(tail(val_counts, val_counts), stored_val['count'], rtol=0, atol=1e-12)
    val_struct = tail(val_structure, val_structure)
    for seed in SEEDS:
        fusion = integrate(stored_val['semantic'], stored_val['count'], val_struct,
                           stored_val[f'temporal_{seed}'])[METHOD]
        np.testing.assert_allclose(np.quantile(fusion, .99), thresholds[str(seed)][METHOD], rtol=0, atol=1e-12)
    window_thresholds = {name: float(np.quantile(values, .99)) for name, values in
                         window_rules(stored_val['semantic'], stored_val['count'], val_struct).items()} if additional_date else {}
    stored_val.close()
    val_actors = np.load(validation_folder / 'actors.npy', mmap_mode='r')
    val_begins = np.r_[0, np.flatnonzero(val_actors[1:] != val_actors[:-1]) + 1]
    sequence_refs = {seed: np.maximum.reduceat(np.load(primary / f'{seed}_block/validation_nll.npy'), val_begins)
                     for seed in SEEDS}

    rows = []
    with closing(EventDay(source, date, split['context'], split['lookback_seconds'])) as day:
        for duration in DURATIONS:
            for fraction in START_FRACTIONS:
                tag = f'all_{duration}s_{int(fraction * 100):02d}'
                folder = target / tag
                report = folder / 'result.json'
                if report.is_file():
                    rows.extend(load(report)['rows'])
                    continue
                offset = int(((test_end - test_start) // NS - duration) * fraction)
                left = test_start + offset * NS
                right = left + duration * NS
                config = dict(
                    mode='synthetic_all_relation_outage', relation_id=None,
                    interval_ns=[left, right], duration_seconds=duration,
                    start_fraction=fraction, start_seconds=offset,
                )
                stream_cache(day, targets, mapping, folder, config)
                metadata = load(folder / 'metadata.json')
                if any(metadata[name] != value for name, value in config.items()):
                    raise ValueError(f'Cache configuration mismatch: {folder}')
                arrays = cache_arrays(folder)
                kept = arrays['indices']
                times = day.events['t'][targets]
                np.testing.assert_array_equal(kept, targets[(times < left) | (times >= right)])
                actors, raw_paths = semantic_scores(
                    source, date, kept, subjects, files, spawn, supports, documents
                )
                raw_count = align(actors, count_scores(folder, count_model, vocabulary))
                struct_ids, struct_features = features(day, kept, mapping, vocabulary)
                np.testing.assert_array_equal(actors, struct_ids)
                paths = semantic_tail(val_paths, raw_paths)
                counts = tail(val_counts, raw_count)
                structure = tail(val_structure, -structure_model.score_samples(struct_features))
                static_scores = dict(Path=paths, Count=raw_count, Structure=structure)
                if additional_date:
                    static_scores.update(window_rules(paths, counts, structure))
                case_rows = []
                for seed, model in models.items():
                    started = time.monotonic()
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    scored, resources = score_cache(model, arrays, device, 1024, folder / f'{seed}_nll.npy')
                    sequence_raw = align(actors, scored)
                    if not np.isfinite(sequence_raw).all():
                        raise ValueError(f'GRU and feature actors differ: {dataset} {tag} {seed}')
                    sequence = tail(sequence_refs[seed], sequence_raw)
                    fusion = integrate(paths, counts, structure, sequence)[METHOD]
                    values = dict(Full=fusion, GRU=sequence_raw, **static_scores)
                    np.savez_compressed(folder / f'{seed}_process_scores.npz', actors=actors, **values)
                    for name, scores in values.items():
                        # Each fixed-score control is deterministic, so record it once per scenario.
                        if name in static_scores and seed != SEEDS[0]:
                            continue
                        threshold = thresholds[str(seed)][METHOD] if name == 'Full' else window_thresholds.get(name)
                        metrics = process_metrics(
                            universe['universe'], universe['positive_processes'],
                            dict(zip(actors.tolist(), scores.tolist())),
                            threshold=threshold,
                        )
                        case_rows.append(dict(dataset=dataset, date=date, case=tag, seed=seed,
                                              method=name, metrics=metrics))
                    save_json(folder / f'{seed}_profile.json', dict(
                        **resources, wall_seconds=time.monotonic()-started, device=str(device),
                        cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0,
                    ))
                    print(f'{dataset} {tag} seed {seed}: {resources["seconds"]:.1f}s', flush=True)
                del arrays
                temporary_report = report.with_suffix('.tmp')
                save_json(temporary_report, dict(metadata=metadata, rows=case_rows))
                temporary_report.replace(report)
                rows.extend(case_rows)
    save_json(target / 'results.json', dict(status='COMPLETED', rows=rows, device=str(device),
                hardware=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                start_fractions=START_FRACTIONS, durations_seconds=DURATIONS,
                window_thresholds=window_thresholds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('theia', 'cadets'), required=True)
    parser.add_argument('--date', help='Evaluate a previously indexed additional campaign window (YYYY-MM-DD)')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = args.output or (ADDITIONAL_OUTPUT if args.date else OUTPUT)
    run(args.dataset, output, torch.device(args.device), additional_date=args.date)


if __name__ == '__main__':
    main()
