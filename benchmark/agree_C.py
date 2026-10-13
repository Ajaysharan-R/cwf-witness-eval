"""Agreement of annotator C with the frozen gold and with A and B. READ-ONLY: writes only the report file.
usage: python benchmark\agree_C.py --c answers_C.txt --split benchmark\data\test3 --a benchmark\annotations_A3.json --b benchmark\annotations_B3.json --out benchmark\agreement_C.txt"""
import argparse, json, os, re, sys

def parse_c(path):
    ans, bad = {}, []
    for ln, line in enumerate(open(path, encoding="utf-8-sig"), 1):
        line = line.strip()
        if not line: continue
        m = re.match(r"^(V\d{2})\s+(.*)$", line)
        if not m: bad.append((ln, line)); continue
        cid, v = m.group(1), m.group(2).strip().lower().replace(" ", "")
        if v == "none": ans[cid] = (True, [])
        elif re.fullmatch(r"p\d+(,p\d+)+(/p\d+(,p\d+)+)*", v):
            ans[cid] = (False, [sorted(w.split(",")) for w in v.split("/")])
        else: bad.append((ln, line))
    return ans, bad

def kappa(x, y):
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    cats = set(x) | set(y); pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def load_ann(path):
    d = json.load(open(path, encoding="utf-8"))["answers"]
    return {k: (v["abstain"], [sorted(w) for w in v["valid_witnesses"]]) for k, v in d.items()}

def load_gold(split):
    out = {}
    for f in sorted(os.listdir(split)):
        if f.endswith(".json") and f.startswith("V"):
            d = json.load(open(os.path.join(split, f), encoding="utf-8"))
            g = d["gold"]; out[d["case_id"]] = (bool(g["abstain"]), [sorted(w) for w in g["valid_witnesses"]], [p["id"] for p in d["passages"]])
    return out

def compare(name, X, Y, ids, pas, w):
    ab = kappa([X[i][0] for i in ids], [Y[i][0] for i in ids])
    xs, ys = [], []
    for i in ids:
        sx = {p for ws in X[i][1] for p in ws}; sy = {p for ws in Y[i][1] for p in ws}
        for p in pas[i]: xs.append(p in sx); ys.append(p in sy)
    pl = kappa(xs, ys)
    acc = sum((X[i][0] and Y[i][0]) or (not X[i][0] and not Y[i][0] and any(a in Y[i][1] for a in X[i][1])) for i in ids)
    dis = [i for i in ids if not ((X[i][0] and Y[i][0]) or (not X[i][0] and not Y[i][0] and any(a in Y[i][1] for a in X[i][1])))]
    w("%s: n=%d  exact-agreement=%d/%d  abstain-kappa=%.3f  passage-kappa=%.3f" % (name, len(ids), len(ids) - len(dis), len(ids), ab, pl))
    flips = [i for i in dis if X[i][0] != Y[i][0]]
    w("   disagreements (%d): %s" % (len(dis), dis)); w("   of which witness-vs-none flips (%d): %s" % (len(flips), flips))
    return dis

ap = argparse.ArgumentParser()
for k in ("c", "split", "a", "b", "out"): ap.add_argument("--" + k, required=True)
a = ap.parse_args(); lines = []; w = lambda s: (print(s), lines.append(s))
C, bad = parse_c(a.c)
gold = load_gold(a.split); ids = sorted(gold); pas = {i: gold[i][2] for i in ids}
missing = [i for i in ids if i not in C]
if bad or missing:
    print("INVALID answers file. Bad lines:", bad[:10], " Missing/blank cases:", missing[:20], "(%d missing)" % len(missing)); sys.exit(1)
G = {i: (gold[i][0], gold[i][1]) for i in ids}
compare("C vs final gold", C, G, ids, pas, w)
compare("C vs A", C, load_ann(a.a), ids, pas, w)
compare("C vs B", C, load_ann(a.b), ids, pas, w)
open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("saved", a.out)
