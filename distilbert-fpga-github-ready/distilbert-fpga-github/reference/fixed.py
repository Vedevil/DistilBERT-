"""Bit-exact, independent integer reference for rtl/*.sv. No torch required."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MAX_HEAD_DIM = 64
SCORE_FRAC = 10
QKV_FRAC = 11
PROB_FRAC = 15
EXP_STEP_BITS = 6  # Q6.10 step 64 => real step 1/16
RECIP_NUMERATOR = 1 << 38
RECIP_SHIFT = 23
EXP_TABLE = tuple(int(math.floor(math.exp(-i / 16.0) * (1 << 15) + 0.5)) for i in range(257))


def saturate_i16(x: int) -> int:
    return min(32767, max(-32768, int(x)))


def round_shift_away(x: int, shift: int) -> int:
    """Round signed integer right shift to nearest, ties away from zero."""
    mag = (abs(int(x)) + (1 << (shift - 1))) >> shift
    return -mag if x < 0 else mag


def quantize_qkv(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mag = np.floor(np.abs(x) * (1 << QKV_FRAC) + 0.5)
    return np.clip(np.copysign(mag, x), -32768, 32767).astype(np.int16)


def score_fixed(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Exactly 64 signed Q4.11 products, scale /8 and round to signed Q6.10."""
    q, k = np.asarray(q), np.asarray(k)
    assert q.ndim == k.ndim == 2 and q.shape[1] == k.shape[1] == MAX_HEAD_DIM
    scores = np.empty((len(q), len(k)), dtype=np.int16)
    for i in range(len(q)):
        for j in range(len(k)):
            dot = sum(int(q[i, d]) * int(k[j, d]) for d in range(MAX_HEAD_DIM))
            scores[i, j] = saturate_i16(round_shift_away(dot, 15))
    return scores


def exp_fixed(diff: int, keep: bool) -> int:
    if not keep or diff >= (16 << SCORE_FRAC):
        return 0
    assert diff >= 0
    idx, frac = divmod(diff, 1 << EXP_STEP_BITS)
    a, b = EXP_TABLE[idx], EXP_TABLE[idx + 1]
    return a - (((a - b) * frac + 32) >> EXP_STEP_BITS)


def softmax_fixed(scores: Iterable[int], keep: Iterable[int]) -> np.ndarray:
    """Q6.10 signed scores -> Q1.15 unsigned probabilities; all-masked -> zeros."""
    xs, flags = [int(i) for i in scores], [bool(i) for i in keep]
    if not xs or len(xs) != len(flags):
        raise ValueError("nonempty, equally sized scores and keep required")
    if not any(flags):
        return np.zeros(len(xs), dtype=np.uint16)
    mx = max(s for s, f in zip(xs, flags) if f)
    exps = [exp_fixed(mx - s, f) for s, f in zip(xs, flags)]
    denom = sum(exps)
    assert denom >= 32768  # at least one unmasked max contributes 1.0
    gain, rem = divmod(RECIP_NUMERATOR, denom)
    gain += int(2 * rem >= denom)
    assert gain < (1 << 24)
    return np.array([min(32768, (e * gain + (1 << 22)) >> RECIP_SHIFT) for e in exps], dtype=np.uint16)


def softmax_float(scores: Iterable[float], keep: Iterable[int]) -> np.ndarray:
    """Stable float64 comparison; masked entries are excluded, not treated as a finite logit."""
    xs = np.asarray(list(scores), dtype=np.float64)
    flags = np.asarray(list(keep), dtype=bool)
    if len(xs) != len(flags) or not len(xs):
        raise ValueError("nonempty, equally sized scores and keep required")
    ans = np.zeros_like(xs)
    if flags.any():
        exps = np.exp(xs[flags] - np.max(xs[flags]))
        ans[flags] = exps / exps.sum()
    return ans


def attention_fixed(q: np.ndarray, k: np.ndarray, v: np.ndarray, keep: Iterable[int]):
    """Pre-projected one-head integer implementation, returns (scores, P, context)."""
    q, k, v = (np.asarray(x, dtype=np.int16) for x in (q, k, v))
    keep = np.asarray(list(keep), dtype=np.uint8)
    assert q.shape == k.shape == v.shape and q.shape[1] == MAX_HEAD_DIM
    assert len(keep) == len(q)
    scores = score_fixed(q, k)
    probs = np.stack([softmax_fixed(row, keep) for row in scores])
    context = np.zeros_like(q)
    for i in range(len(q)):
        for d in range(MAX_HEAD_DIM):
            acc = sum(int(probs[i, j]) * int(v[j, d]) for j in range(len(k)))
            context[i, d] = saturate_i16(round_shift_away(acc, PROB_FRAC))
    return scores, probs, context
