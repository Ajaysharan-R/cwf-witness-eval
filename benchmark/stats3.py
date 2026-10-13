"""Full metrics + statistics for every method on one split (use on test3 AFTER all baselines are fixed).
    python benchmark\\stats3.py --split benchmark\\data\\test3 --nli benchmark\\nli_test3.json --llama benchmark\\llama_test3.json ^
        --llama_abs benchmark\\llama_abstain_test3.json --ref cwf_v21 --out benchmark\\stats_test3.json
Reports: overall accuracy (Wilson CI), answerable exact, passage P/R/F1 (answerable), abstention P/R/F1 (abstain = positive),
false-abstention rate, median runtime ms, exact McNemar vs --ref (overall and abstain subgroup) with Holm correction,
paired-bootstrap accuracy difference CI. Nothing is tuned here."""
from __future__ import annotations
import argparse, json, math, os, random, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from benchmark.schema import load_split
from benchmark import baselines as B
from benchmark.metrics import score, mcnemar_exact


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def boot_diff(a, b, n_boot=10000, seed=0):
    rng = random.Random(seed); n = len(a); d = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] - b[i] for i in idx) / n)
    d.sort(); return d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--nli"); ap.add_argument("--llama"); ap.add_argument("--llama_abs")
    ap.add_argument("--extra", action="append", default=[], help="NAME=path of an extra cached LLM selection file (repeatable); timings read from path.times.json")
    ap.add_argument("--ref", default="cwf_v21"); ap.add_argument("--out"); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    cases = load_split(a.split); n = len(cases)
    M = {"cwf_v1": B.cwf, "cwf_v2": B.cwf_v2, "cwf_v21": B.cwf_v21, "no_context_filter": B.no_context_filter,
         "bm25_top2": B.bm25_top2, "full_evidence": B.full_evidence}
    if a.nli and os.path.exists(a.nli): M["nli_top2"] = B.make_nli_top2(a.nli)
    if a.llama and os.path.exists(a.llama): M["llama_selector"] = B.make_cached_selector(a.llama)
    if a.llama_abs and os.path.exists(a.llama_abs): M["llama_abstain"] = B.make_cached_selector(a.llama_abs)
    extra_times = {}
    for e in a.extra:
        nm, pth = e.split("=", 1)
        if os.path.exists(pth):
            M[nm] = B.make_cached_selector(pth)
            if os.path.exists(pth + ".times.json"): extra_times[nm] = json.load(open(pth + ".times.json"))
    th = os.path.join(HERE, "thresholds.json")
    if os.path.exists(th):
        from benchmark import abstain_baselines as AB
        T = json.load(open(th)); M["bm25_abstain"] = AB.make_bm25_abstain(T["bm25_abstain"])
        if a.nli and os.path.exists(a.nli): M["nli_abstain"] = AB.make_nli_abstain(a.nli, T["nli_abstain"])
    cached = {"nli_top2", "nli_abstain", "llama_selector", "llama_abstain"} | set(extra_times) | {e.split("=", 1)[0] for e in a.extra}
    times_llm = {}
    if a.llama_abs and os.path.exists(a.llama_abs + ".times.json"):
        times_llm = json.load(open(a.llama_abs + ".times.json"))
    ans_idx = [i for i, c in enumerate(cases) if not c.abstain]; abs_idx = [i for i, c in enumerate(cases) if c.abstain]
    R = {}
    for m, fn in M.items():
        rows, preds, ms = [], [], []
        for c in cases:
            cef = c.to_cef(); ts = []
            for _ in range(a.reps if m not in cached else 1):
                t0 = time.perf_counter(); pr = fn(c, cef); ts.append((time.perf_counter() - t0) * 1000)
            preds.append(set(pr)); rows.append(score(pr, c)); ms.append(statistics.median(ts))
        ex = [int(r["exact"]) for r in rows]
        tp = sum(1 for i in abs_idx if not preds[i]); fp = sum(1 for i in ans_idx if not preds[i]); fn_ = sum(1 for i in abs_idx if preds[i])
        pr_ = tp / (tp + fp) if tp + fp else float("nan"); rc_ = tp / (tp + fn_) if tp + fn_ else float("nan")
        f1 = 2 * pr_ * rc_ / (pr_ + rc_) if pr_ == pr_ and rc_ == rc_ and pr_ + rc_ else float("nan")
        an = [rows[i] for i in ans_idx]
        runtime = ("%.2f" % statistics.median(ms)) if m not in cached else "n/a (cached)"
        if m == "llama_abstain" and times_llm: runtime = "%.0f (Ollama, measured)" % statistics.median(times_llm.values())
        if m in extra_times: runtime = "%.0f (Ollama, measured)" % statistics.median(extra_times[m].values())
        R[m] = {"ex": ex, "overall": sum(ex), "ans_exact": sum(ex[i] for i in ans_idx), "abs_ok": sum(ex[i] for i in abs_idx),
                "P": sum(r["p"] for r in an) / len(an), "R": sum(r["r"] for r in an) / len(an), "F1": sum(r["f1"] for r in an) / len(an),
                "abst_P": pr_, "abst_R": rc_, "abst_F1": f1, "false_abst": fp / len(ans_idx) if ans_idx else float("nan"), "runtime_ms": runtime}
    L = []; w = lambda s="": (print(s), L.append(s))
    w("split=%s n=%d answerable=%d abstain=%d reference=%s" % (a.split, n, len(ans_idx), len(abs_idx), a.ref))
    w("\n%-18s %-22s %-18s %-16s %-20s %-8s %-8s" % ("method", "overall (Wilson 95%)", "answerable exact", "passage P/R/F1", "abstain P/R/F1", "false-abst", "ms/case"))
    f = lambda x: "%.2f" % x if x == x else " -- "
    for m, r in R.items():
        lo, hi = wilson(r["overall"], n); lo2, hi2 = wilson(r["ans_exact"], len(ans_idx))
        w("%-18s %2d/%d %.2f [%.2f,%.2f]  %2d/%d [%.2f,%.2f]  %s/%s/%s   %s/%s/%s   %s   %s" % (
            m, r["overall"], n, r["overall"] / n, lo, hi, r["ans_exact"], len(ans_idx), lo2, hi2, f(r["P"]), f(r["R"]), f(r["F1"]),
            f(r["abst_P"]), f(r["abst_R"]), f(r["abst_F1"]), f(r["false_abst"]), r["runtime_ms"]))
    ref = R[a.ref]["ex"]; others = [m for m in R if m not in (a.ref, "full_evidence")]  # full_evidence is a trivial floor; excluded from the Holm family
    po = [mcnemar_exact(ref, R[m]["ex"])["p"] for m in others]
    pa = [mcnemar_exact([ref[i] for i in abs_idx], [R[m]["ex"][i] for i in abs_idx])["p"] for m in others]
    ho, ha = holm(po), holm(pa)
    w("\nComparison vs %s (exact McNemar; Holm across the %d comparisons in each column; paired bootstrap 10,000)" % (a.ref, len(others)))
    w("%-18s %-26s %-12s %-12s %-26s %-12s %-12s" % ("method", "overall ref-only/other-only", "p", "p(Holm)", "abstain-subgroup", "p", "p(Holm)"))
    for j, m in enumerate(others):
        mo = mcnemar_exact(ref, R[m]["ex"]); ma = mcnemar_exact([ref[i] for i in abs_idx], [R[m]["ex"][i] for i in abs_idx])
        lo, hi = boot_diff(ref, R[m]["ex"])
        w("%-18s %2d / %-2d  diff CI [%+.3f,%+.3f]   %-12.4f %-12.4f %2d / %-2d              %-12.4f %-12.4f" % (
            m, mo["a_only"], mo["b_only"], lo, hi, mo["p"], ho[j], ma["a_only"], ma["b_only"], ma["p"], ha[j]))
    w("\nThe abstain-subgroup column is a subgroup analysis (16 cases), not an overall-superiority test.")
    if a.out:
        json.dump({m: {k: v for k, v in r.items()} for m, r in R.items()}, open(a.out, "w"), indent=1)
        open(a.out.replace(".json", ".txt"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("saved", a.out)


if __name__ == "__main__":
    main()
