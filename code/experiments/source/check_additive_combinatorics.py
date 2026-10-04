"""Bounded independent exhaustive-pair audit of the additive selector.

No physical LP is solved, no data is read, and no accuracy baseline is run.
All comparisons use exact additive cost arithmetic, including binary floats.
"""
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from additive_selection import select_from_catalog, threshold_minimum


def exact(x):
    return x if isinstance(x, int) else Fraction(x)


def independent_entries(catalog, costs):
    c = [exact(x) for x in costs]
    return [dict(ids=tuple(r['ids']), positive=r['positive'], negative=r['negative'],
                 cost=sum(c[i] for i in r['ids'])) for r in catalog], c


def full_pair_optimum(catalog, costs, budget=None, threshold=None):
    """Explicit independent Cartesian product, with no reduced families."""
    best = None
    for p in catalog:
        for q in catalog:
            score = max(p['positive'], q['negative'])
            if threshold is not None and score > threshold:
                continue
            ids = tuple(sorted(set(p['ids']).union(q['ids'])))
            cost = sum(costs[i] for i in ids)
            if budget is not None and cost > exact(budget):
                continue
            rank = ((cost, len(ids), ids) if threshold is not None
                    else (score, cost, len(ids), ids))
            if best is None or rank < best:
                best = rank
    return best


def row(ids, positive, negative):
    return dict(ids=ids, positive=positive, negative=negative)


def cases():
    base_budgets = [0, 1, 2, 3, 4, 7, 20]
    yield 'empty_catalog', [], [], [0, 1]
    yield 'empty_certificate', [row((), 2, 3)], [0, 1], [0, 1]
    yield 'no_empty_and_unaffordable', [row((0,), 1, 1)], [3], [0, 2, 3]
    yield 'singleton_opposite_directions', [row((0,), 0, 9), row((1,), 9, 0)], [1, 2], base_budgets
    yield 'double_overlap_must_be_checked', [row((0, 1), 0, 1), row((0, 2), 9, 1), row((1, 2), 9, 1)], [1, 1, .5], [0, 1, 2, 2.5, 3]
    yield 'all_zero_cost_full_ties', [row(s, 0, 0) for k in range(3) for s in combinations(range(4), k)], [0]*4, [0, 1]
    yield 'zero_cost_with_cardinality_ties', [row((0, 1), 0, 2), row((0, 2), 2, 0), row((1,), 0, 2), row((2,), 2, 0), row((3,), 0, 0)], [0, 0, 0, 0], [0, 1]
    yield 'lexicographic_fixed_disjoint_union', [row((1, 4), 0, 9), row((0, 5), 9, 0), row((2, 3), 9, 0)], [1]*6, [0, 2, 3, 4, 6]
    yield 'lexicographic_overlap_one', [row((1, 4), 0, 9), row((0, 1), 9, 0), row((1, 2), 9, 0), row((3, 4), 9, 0)], [1]*5, [0, 2, 3, 4, 5]
    yield 'avoidance_tree_depth_two', [row((0, 2), 0, 9), row((0, 1), 9, 0), row((2, 3), 9, 0), row((4, 5), 9, 0)], [0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5, 8]
    yield 'floating_prices_exact_binary', [row((), 9, 9), row((0,), 0, 9), row((1,), 9, 0), row((2,), 1, 1)], [.1, .2, .3], [0, .1, .2, .3, .30000000000000004, .4, .6, 1]
    yield 'mixed_empty_single_double_families', [row((), 2, 8), row((0,), 5, 0), row((1, 2), 0, 4), row((0, 3), 1, 1)], [0, 2, 1, 0], base_budgets
    # Small predeclared random instances supplement, rather than replace,
    # the deliberate edge constructions. Scores need not come from LPs.
    for seed in range(12):
        rng = random.Random(87000 + seed)
        n = 1 + seed % 7
        subsets = [s for k in range(3) for s in combinations(range(n), k)]
        if seed % 2:
            subsets = [s for s in subsets if rng.random() < .7]
        catalog = [row(s, rng.randrange(4), rng.randrange(4)) for s in subsets]
        costs = [rng.randrange(4) for _ in range(n)]
        yield f'seeded_small_{seed:02d}', catalog, costs, base_budgets


def main():
    records = []
    budget_checks = threshold_checks = 0
    for name, catalog, raw_costs, budgets in cases():
        entries, costs = independent_entries(catalog, raw_costs)
        levels = sorted({r[k] for r in catalog for k in ('positive', 'negative')})
        # Include an empty-family threshold when nonnegative scores permit it.
        levels = [-1] + levels
        rec = dict(name=name, catalog=catalog, costs=raw_costs,
                   cost_exact=[str(x) for x in costs], budgets=budgets,
                   thresholds=levels, budget_results=[], threshold_results=[])
        for tau in levels:
            expected = full_pair_optimum(catalog, costs, threshold=tau)
            actual = threshold_minimum(entries, costs, tau)
            got = None if actual is None else (actual['cost'], len(actual['anchor_ids']), actual['anchor_ids'])
            assert got == expected, (name, 'threshold', tau, expected, got)
            if actual is not None:
                assert actual['representative_count'] <= 7
                assert actual['negative_scans'] <= 7
                assert actual['candidate_checks'] <= 13 * actual['eligible_positive']
            rec['threshold_results'].append(dict(threshold=tau, optimum=None if got is None else [str(got[0]), got[1], got[2]]))
            threshold_checks += 1
        for budget in budgets:
            expected = full_pair_optimum(catalog, costs, budget=budget)
            actual = select_from_catalog(catalog, raw_costs, budget)
            got = None if not actual['feasible'] else (actual['selection_bound'], actual['cost'], len(actual['anchor_ids']), actual['anchor_ids'])
            assert got == expected, (name, 'budget', budget, expected, got)
            if actual['feasible']:
                lookup = {tuple(r['ids']):r for r in catalog}
                p, q = actual['positive_support'], actual['negative_support']
                assert tuple(sorted(set(p).union(q))) == actual['anchor_ids']
                assert max(lookup[p]['positive'], lookup[q]['negative']) == actual['selection_bound']
                assert sum(costs[i] for i in actual['anchor_ids']) == actual['cost'] <= exact(budget)
            rec['budget_results'].append(dict(budget=budget, optimum=None if got is None else [got[0], str(got[1]), got[2], got[3]]))
            budget_checks += 1
        records.append(rec)
    target = HERE / 'additive_selection.py'
    report = dict(status='PASS', scope='bounded pure-combinatorial exhaustive-pair comparisons; no physical LP or data experiment',
                  implementation_sha256=sha256(target.read_bytes()).hexdigest(),
                  checker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  cases=len(records), budget_checks=budget_checks, threshold_checks=threshold_checks,
                  all_comparisons_use_exact_additive_costs=True,
                  oracle='independent Cartesian product over every positive/negative certificate pair',
                  tie_rule=['directional_max_score', 'union_additive_cost', 'union_size', 'sorted_union_ids'],
                  records=records)
    output = HERE / 'results/additive_combinatorics_check.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
