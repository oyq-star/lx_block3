"""Diagnostic NLL comparison on label-blind paired targets using a frozen GRU."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from causal_data import EventDay, draw_contexts, encode_positions
from train_eval import EventGRU, ROOT, padded, seed_everything


MODES = ('clean', 'iid', 'dispersed', 'block')


def paired_summary(left, right, effective, distinct):
    differences = left.astype(np.float64)-right.astype(np.float64)
    return {name: dict(n=len(values), mean=float(np.mean(values)),
                       median=float(np.median(values)),
                       fraction_positive=float(np.mean(values > 1e-6))) if len(values) else dict(n=0)
            for name, values in [('all', differences), ('effective', differences[effective]),
                                 ('effective_distinct_input', differences[effective & distinct])]}


def score_view(model, sequences, targets, context, device):
    tokens, lengths = padded(sequences, context)
    with torch.no_grad():
        return F.cross_entropy(model(torch.from_numpy(tokens).to(device),
                                    torch.from_numpy(lengths).to(device)), targets,
                               reduction='none').cpu().numpy()


def build_view(day, indices, token_map, duration, mask_seed):
    sequences = {mode: [] for mode in MODES}
    k_raw = []
    k_eff = []
    for index in indices:
        original = None
        for mode in MODES[1:]:
            clean, corrupt, summary = draw_contexts(day, int(index), mode, mask_seed, 0, (duration,), 0)
            counts = (summary['K_raw'], summary['K_eff'])
            if original is None:
                original = counts
                sequences['clean'].append(encode_positions(day, clean, token_map))
            if counts != original:
                raise RuntimeError('Masks disagree on deletion count')
            sequences[mode].append(encode_positions(day, corrupt, token_map))
        k_raw.append(original[0])
        k_eff.append(original[1])
    return sequences, k_raw, k_eff


def evaluate(day, indices, token_map, duration, mask_seed, model, device, deadline):
    nll = {mode: [] for mode in MODES}
    distinct = {mode: [] for mode in ('iid', 'dispersed', 'clean')}
    k_raw, k_eff = [], []
    for offset in range(0, len(indices), 128):
        if time.monotonic() >= deadline:
            raise TimeoutError('Paired diagnostic deadline reached')
        rows = indices[offset:offset+128]
        sequences, raw, effective = build_view(day, rows, token_map, duration, mask_seed)
        k_raw.extend(raw)
        k_eff.extend(effective)
        targets = torch.from_numpy(token_map[day.raw_tokens(rows)].astype(np.int64)).to(device)
        for mode, tokens in sequences.items():
            nll[mode].extend(score_view(model, tokens, targets, day.k, device))
        for mode in distinct:
            distinct[mode].extend(not np.array_equal(left, right)
                                  for left, right in zip(sequences['block'], sequences[mode]))
    arrays = {mode: np.asarray(values) for mode, values in nll.items()}
    effective = np.asarray(k_eff) > 0
    for mode, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite NLL: '+mode)
    return dict(targets=len(indices), K_raw_total=sum(k_raw), K_eff_total=sum(k_eff),
                effective_count=int(effective.sum()), block_distinct_input_counts={mode: int(sum(flags)) for mode, flags in distinct.items()},
                mean_NLL={mode: float(values.mean()) for mode, values in arrays.items()},
                block_minus={mode: paired_summary(arrays['block'], arrays[mode], effective, np.asarray(flags))
                             for mode, flags in distinct.items()}), arrays, np.asarray(k_raw), np.asarray(k_eff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--seconds', type=int, default=600)
    args = parser.parse_args()
    started = time.monotonic()
    split = json.loads((args.cache / 'split.json').read_text(encoding='utf-8'))
    deadline = started + args.seconds
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        raise RuntimeError('Refusing to overwrite completed results')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    if config['dataset']['split'] != split or config['mode'] != 'clean':
        raise RuntimeError('Expected the matching clean checkpoint')
    model = EventGRU(config['dataset']['vocabulary_size'], config['embedding'], config['hidden']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    token_map = np.load(args.cache / 'token_map.npy')
    observations = {}
    for partition, date, folder in [('validation', split['validation_date'], 'validation'),
                                     ('test', split['test_date'], 'test_clean')]:
        candidates = np.load(args.cache / folder / 'indices.npy', mmap_mode='r')
        count = min(args.samples, len(candidates))
        sampled = np.sort(np.random.RandomState(args.seed).choice(candidates, size=count, replace=False))
        day = EventDay(ROOT / ('data/processed/'+split['dataset']+'_e3'), date, split['context'], split['lookback_seconds'])
        try:
            stats, scores, k_raw, k_eff = evaluate(day, sampled, token_map, split['duration_seconds'],
                                                    split['mask_seed'], model, device, deadline)
            observations[partition] = stats
            np.savez(args.output / (partition+'_paired.npz'), indices=sampled, K_raw=k_raw, K_eff=k_eff, **scores)
            print(json.dumps(dict(partition=partition, **stats)), flush=True)
        finally:
            day.close()
    result = dict(scope='label-blind target sample; diagnostic NLL only, no process AP or full-stream claims',
                  status='COMPLETED', seed=args.seed, split=split, checkpoint=str(args.checkpoint),
                  elapsed_seconds=time.monotonic()-started, observations=observations)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
