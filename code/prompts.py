"""Deterministic prompt generators for the four ICL templates.

All four tasks share the 4-shot surface form ``f0,f1,Y \n ... f0q,f1q,``.
Hierarchical and modular have an integer feature pair drawn from
``{0..7}^2``; antonym and country-capital have token pairs from a fixed
vocabulary list.

Prompts are returned as ``dict`` with keys ``prompt``, ``correct``,
``incorrect``, ``rule`` (where applicable).
"""
from __future__ import annotations
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Hierarchical & modular
# ---------------------------------------------------------------------------
HIER_RULES = (
    lambda f0, f1: f0 > 3,
    lambda f0, f1: f1 > 3,
    lambda f0, f1: (f0 + f1) > 7,
    lambda f0, f1: f0 == f1,
)
MOD_DIVISORS = (2, 3, 5, 7)
LABEL_TOKENS = ("A", "B")


def _format_rule_prompt(pairs, query, label_fn) -> dict:
    lines = [f"{f0},{f1},{LABEL_TOKENS[int(label_fn(f0, f1))]}" for f0, f1 in pairs]
    correct = LABEL_TOKENS[int(label_fn(*query))]
    incorrect = LABEL_TOKENS[1 - int(label_fn(*query))]
    return {
        "prompt": "\n".join(lines) + f"\n{query[0]},{query[1]},",
        "correct": correct,
        "incorrect": incorrect,
    }


def make_hierarchical(seed: int, n_demos: int = 4) -> dict:
    rng = random.Random(seed)
    rule_idx = rng.randrange(len(HIER_RULES))
    rule = HIER_RULES[rule_idx]
    pairs = []
    for _ in range(n_demos):
        f0, f1 = rng.randint(0, 7), rng.randint(0, 7)
        pairs.append((f0, f1))
    query = (rng.randint(0, 7), rng.randint(0, 7))
    out = _format_rule_prompt(pairs, query, rule)
    out["task_id"] = "hierarchical"
    out["rule_idx"] = rule_idx
    return out


def make_modular(seed: int, n_demos: int = 4) -> dict:
    rng = random.Random(seed)
    m = rng.choice(MOD_DIVISORS)
    pairs = []
    for _ in range(n_demos):
        f0, f1 = rng.randint(0, 7), rng.randint(0, 7)
        pairs.append((f0, f1))
    query = (rng.randint(0, 7), rng.randint(0, 7))
    out = _format_rule_prompt(pairs, query, lambda a, b: ((a + b) % m) == 0)
    out["task_id"] = "modular"
    out["m"] = m
    return out


# ---------------------------------------------------------------------------
# Vocabulary ICL: antonym, country-capital
# ---------------------------------------------------------------------------
ANTONYM_PAIRS = [
    ("hot", "cold"), ("up", "down"), ("happy", "sad"), ("rich", "poor"),
    ("fast", "slow"), ("big", "small"), ("hard", "soft"), ("wet", "dry"),
    ("strong", "weak"), ("light", "dark"), ("wide", "narrow"),
    ("clean", "dirty"), ("full", "empty"), ("near", "far"),
    ("alive", "dead"), ("loud", "quiet"), ("smooth", "rough"),
    ("thick", "thin"), ("young", "old"), ("warm", "cool"),
    # ... (complete pair list shipped as JSON in code/data/antonym_pairs.json)
]

COUNTRY_CAPITAL = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"),
    ("Spain", "Madrid"), ("Italy", "Rome"), ("Russia", "Moscow"),
    ("China", "Beijing"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
    ("Brazil", "Brasilia"),
    # ... (complete list shipped as JSON in code/data/country_capital.json)
]


def _format_vocab(pairs, query_pair) -> dict:
    lines = [f"{a}:{b}" for a, b in pairs]
    return {
        "prompt": " ".join(lines) + f" {query_pair[0]}:",
        "correct": query_pair[1],
        "incorrect": pairs[0][1] if pairs[0][1] != query_pair[1] else pairs[1][1],
    }


def make_vocab(task: str, seed: int, n_demos: int = 4) -> dict:
    pool = ANTONYM_PAIRS if task == "antonym" else COUNTRY_CAPITAL
    rng = random.Random(seed)
    samp = rng.sample(pool, n_demos + 1)
    demos, query = samp[:-1], samp[-1]
    out = _format_vocab(demos, query)
    out["task_id"] = task
    return out


# ---------------------------------------------------------------------------
# Batched generators
# ---------------------------------------------------------------------------
def generate_batch(task_id: str, n: int, base_seed: int) -> list[dict]:
    """Generate ``n`` independent prompts for the named task."""
    if task_id == "hierarchical":
        return [make_hierarchical(base_seed + i) for i in range(n)]
    if task_id == "modular":
        return [make_modular(base_seed + i) for i in range(n)]
    if task_id in ("antonym", "country-capital"):
        return [make_vocab(task_id, base_seed + i) for i in range(n)]
    raise ValueError(f"unknown task_id: {task_id}")


def make_paired_rule_flip(task_id: str, seed: int) -> tuple[dict, dict]:
    """Return ``(correct_prompt, rule_flipped_prompt)`` differing only in
    which hidden rule generated the in-context demonstrations.

    Used for the per-prompt ``swing`` measure
    :math:`\\Delta\\ell(x_c) - \\Delta\\ell(x_r)`.
    """
    if task_id == "hierarchical":
        rng = random.Random(seed)
        idx = rng.randrange(len(HIER_RULES))
        flip_idx = (idx + 1 + rng.randrange(len(HIER_RULES) - 1)) % len(HIER_RULES)
        # generate same demos, two different rules
        pairs = [(rng.randint(0, 7), rng.randint(0, 7)) for _ in range(4)]
        query = (rng.randint(0, 7), rng.randint(0, 7))
        return (
            _format_rule_prompt(pairs, query, HIER_RULES[idx])      | {"task_id": "hierarchical", "rule_idx": idx},
            _format_rule_prompt(pairs, query, HIER_RULES[flip_idx]) | {"task_id": "hierarchical", "rule_idx": flip_idx},
        )
    if task_id == "modular":
        rng = random.Random(seed)
        m1, m2 = rng.sample(MOD_DIVISORS, 2)
        pairs = [(rng.randint(0, 7), rng.randint(0, 7)) for _ in range(4)]
        query = (rng.randint(0, 7), rng.randint(0, 7))
        return (
            _format_rule_prompt(pairs, query, lambda a, b, m=m1: ((a + b) % m) == 0) | {"task_id": "modular", "m": m1},
            _format_rule_prompt(pairs, query, lambda a, b, m=m2: ((a + b) % m) == 0) | {"task_id": "modular", "m": m2},
        )
    raise ValueError(f"rule-flip pairs only defined for hierarchical / modular; got {task_id}")
