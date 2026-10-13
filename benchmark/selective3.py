"""Selective-prediction curves: accuracy on ANSWERED cases vs coverage, sweeping the abstention threshold of BM25 and NLI.
Nothing is tuned here (the curve is the sweep); the thresholds tuned on test v1 are marked. CWF variants are single points.
    python benchmark\\selective3.py --split benchmark\\data\\test3 --nli benchmark\\nli_test3.json --out benchmark\\selective_test3"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from benchmark.schema import load_split
from benchmark import baselines as B, abstain_baselines as AB
from benchmark.metrics import score

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--nli"); ap.add_argument("--out", required=True)
a = ap.parse_args(); cases = load_split(a.split); n = len(cases)
T = json.load(open(os.path.join(HERE, "thresholds.json")))


def curve(feat, top2):
    f = [feat(c) for c in cases]; ex = [int(score(top2(c), c)["exact"]) for c in cases]; pts = []
    for th in sorted(set(f)) + [float("inf")]:
        ans = [i for i in range(n) if f[i] >= th]
        pts.append((th, len(ans) / n, (sum(ex[i] for i in ans) / len(ans)) if ans else float("nan")))
    return pts


def point(fn):
    pr = [set(fn(c, c.to_cef())) for c in cases]; ans = [i for i in range(n) if pr[i]]
    return len(ans) / n, (sum(int(score(pr[i], cases[i])["exact"]) for i in ans) / len(ans)) if ans else float("nan")


curves = {"BM25-tau": curve(AB.bm25_feature, lambda c: set(p for _, p in AB.bm25_scores(c)[:2]))}
tuned = {"BM25-tau": T["bm25_abstain"]}
if a.nli and os.path.exists(a.nli):
    nd = json.load(open(a.nli, encoding="utf-8"))
    curves["NLI-tau"] = curve(lambda c: AB.nli_feature(c, nd), lambda c: set(k for k, _ in sorted(nd[c.case_id].items(), key=lambda kv: (-kv[1], kv[0]))[:2]))
    tuned["NLI-tau"] = T["nli_abstain"]
pts = {"CWF v1": point(B.cwf), "CWF v2.1": point(B.cwf_v21)}
with open(a.out + ".csv", "w") as f:
    f.write("series,threshold,coverage,selective_accuracy\n")
    for k, v in curves.items():
        for th, cv, ac in v: f.write("%s,%s,%.4f,%.4f\n" % (k, th, cv, ac))
    for k, (cv, ac) in pts.items(): f.write("%s,point,%.4f,%.4f\n" % (k, cv, ac))
print("saved", a.out + ".csv"); [print("  %s: coverage %.2f, accuracy on answered %.2f" % (k, *v)) for k, v in pts.items()]
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)
    for (k, v), mk in zip(curves.items(), ("-", "--")):
        xs = [p[1] for p in v if p[2] == p[2]]; ys = [p[2] for p in v if p[2] == p[2]]; ax.plot(xs, ys, mk, lw=1.2, label=k)
    for (k, (cv, ac)), mk in zip(pts.items(), ("s", "*")):
        ax.plot([cv], [ac], mk, ms=8, label=k)
    ax.set_xlabel("Coverage (fraction of cases answered)"); ax.set_ylabel("Exact match on answered cases"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3); ax.legend(fontsize=6, loc="lower left"); fig.tight_layout(); fig.savefig(a.out + ".png"); print("saved", a.out + ".png")
except ImportError:
    print("matplotlib not installed: use the CSV (pip install matplotlib to also get the PNG)")
