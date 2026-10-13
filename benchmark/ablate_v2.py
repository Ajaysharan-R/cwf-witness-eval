"""Drop-one rule ablation of CWF v2 (v2.1 predictions are identical on test3). POST HOC ANALYSIS of a frozen method on a split:
nothing is tuned or changed in the method files (rules are switched off by monkey-patching in memory only). Label it post hoc.
    python benchmark\\ablate_v2.py --split benchmark\\data\\test3"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cwf")); sys.path.insert(0, HERE)
from benchmark.schema import load_split
from benchmark.metrics import score
import witness_v2 as w2
from fingerprint import ConflictAttribute

ORIG = {k: getattr(w2, k) for k in ("NEG", "UNIT_MG", "_window", "_entity_ok", "_pair_ok")}
FLAGS = {"pop": True, "jur": True, "spec": True, "tail": True, "role": True}


def pair_ok(cef, p, q, strict=True):
    if FLAGS["pop"] and strict and not w2.pop_compatible(p.population, q.population): return False
    if FLAGS["jur"] and strict and cef.attribute != ConflictAttribute.JURISDICTION and p.jurisdiction and q.jurisdiction \
            and p.jurisdiction.strip().lower() != q.jurisdiction.strip().lower(): return False
    if FLAGS["spec"] and strict and cef.attribute != ConflictAttribute.SPECIALTY and p.specialty and q.specialty \
            and p.specialty.strip().lower() != q.specialty.strip().lower(): return False
    if not strict: return True
    if FLAGS["tail"]:
        (ta, ka), (tb, kb) = w2._tail(cef, p), w2._tail(cef, q)
        if ta and tb and ka == kb and not (ta <= tb or tb <= ta): return False
    if FLAGS["role"]:
        rp, rq = w2._role(p), w2._role(q)
        if rp and rq and rp != rq: return False
    return True


def reset():
    for k, v in ORIG.items(): setattr(w2, k, v)
    for k in FLAGS: FLAGS[k] = True
    w2._pair_ok = pair_ok


def run(cases):
    out = []
    for c in cases:
        r = w2.generate_minimal_witness_v2(c.to_cef())
        out.append(set(r.witness.passage_ids) if r.found else set())
    return out


ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); a = ap.parse_args()
cases = load_split(a.split); n = len(cases)
w2._pair_ok = ORIG["_pair_ok"]; real = run(cases)           # untouched frozen v2
reset(); full = run(cases)
assert real == full, "ablation harness does not reproduce frozen v2 -- do not use"
print("harness check: all rules on == frozen v2 predictions on all %d cases\n" % n)


def tab(name, preds):
    sc = [score(p, c) for p, c in zip(preds, cases)]
    return name, sum(int(s["exact"]) for s in sc), sum(int(s["exact"]) for s, c in zip(sc, cases) if not c.abstain), sum(int(s["exact"]) for s, c in zip(sc, cases) if c.abstain)


rows = [tab("full v2 (all rules)", full)]
def ablate(name, fn):
    reset(); fn(); rows.append(tab(name, run(cases)))
ablate("- negation handling (negations = ordinary words)", lambda: setattr(w2, "NEG", set()))
ablate("- unit normalisation", lambda: setattr(w2, "UNIT_MG", {}))
ablate("- time window", lambda: setattr(w2, "_window", lambda tc: None))
ablate("- other-drug (entity) check", lambda: setattr(w2, "_entity_ok", lambda cef, p: True))
for k, lab in (("pop", "population"), ("jur", "jurisdiction"), ("spec", "specialty"), ("tail", "clinical-situation tail"), ("role", "dose role")):
    ablate("- pairwise %s check" % lab, lambda k=k: FLAGS.__setitem__(k, False))
ablate("- ALL pairwise checks (= legacy)", lambda: [FLAGS.__setitem__(k, False) for k in FLAGS])
print("%-50s %9s %11s %12s %8s" % ("variant", "overall", "answerable", "abstain-ok", "delta"))
base = rows[0][1]
for name, o, an, ab in rows:
    print("%-50s %4d/%-4d %6d/%-4d %7d/%-4d %+6d" % (name, o, n, an, sum(1 for c in cases if not c.abstain), ab, sum(1 for c in cases if c.abstain), o - base))
reset()
