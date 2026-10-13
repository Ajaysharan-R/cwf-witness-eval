"""ORACLE-threshold analysis for the abstaining baselines. The threshold is chosen ON THE EVALUATION SPLIT, so this is an
upper bound for BM25-tau / NLI-tau, NOT a valid baseline. Report it next to (never instead of) the pre-tuned rows.
    python benchmark\\oracle_tau.py --split benchmark\\data\\test3 --nli benchmark\\nli_test3.json"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from benchmark.schema import load_split
from benchmark import abstain_baselines as AB
from benchmark.metrics import score

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--nli")
a = ap.parse_args(); cases = load_split(a.split); n = len(cases)
na = sum(1 for c in cases if c.abstain)


def sweep(name, feat, make):
    f = [feat(c) for c in cases]; res = []
    for th in sorted(set(f)) + [float("inf")]:
        fn = make(th); sc = [score(fn(c, c.to_cef()), c) for c in cases]
        res.append((sum(int(s["exact"]) for s in sc), sum(int(s["exact"]) for s, c in zip(sc, cases) if c.abstain), th))
    best = max(r[0] for r in res); top = [r for r in res if r[0] == best]
    ths = [r[2] for r in top if r[2] != float("inf")]
    print("%-9s oracle best overall %d/%d (abstain-correct %s/%d) at threshold in [%.4f, %.4f]  (tuned-on-v1 row for reference is in stats_test3.txt)" % (
        name, best, n, sorted(set(r[1] for r in top)), na, min(ths) if ths else float("inf"), max(ths) if ths else float("inf")))


sweep("BM25-tau", AB.bm25_feature, AB.make_bm25_abstain)
if a.nli and os.path.exists(a.nli):
    nd = json.load(open(a.nli, encoding="utf-8"))
    sweep("NLI-tau", lambda c: AB.nli_feature(c, nd), lambda th: AB.make_nli_abstain(a.nli, th))
print("Interpretation: if CWF v2.1 (49/60, 11/16 abstain-correct) still matches or exceeds these upper bounds, threshold tuning cannot explain its abstention advantage.")
