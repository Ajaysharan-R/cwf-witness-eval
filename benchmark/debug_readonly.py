"""READ-ONLY diagnostic: shows which v2.1 stage rejects each pair for the given cases. Changes nothing."""
import sys, os, json, itertools
ROOT = os.environ.get("CWF_ROOT", r".")
sys.path.insert(0, os.path.join(ROOT, "benchmark")); sys.path.insert(0, os.path.join(ROOT, "cwf"))
import schema, witness_v2 as w2, witness_v21 as w21
split = sys.argv[1]; ids = sys.argv[2:]
for cid in ids:
    d = json.load(open(os.path.join(split, cid + ".json"), encoding="utf-8"))
    c = schema.from_dict(d); cef = c.to_cef()
    print("=" * 70); print(cid, "entity=%r value_a=%r value_b=%r attr=%s" % (c.entity, c.value_a, c.value_b, c.attribute))
    print("parsed value_a ->", w2._parse_qty(c.value_a), "| value_b ->", w2._parse_qty(c.value_b))
    A = w21.side_ids(cef, cef.value_a, cef.value_b); B = w21.side_ids(cef, cef.value_b, cef.value_a)
    print("side A passages:", sorted(A), " side B passages:", sorted(B))
    for p in cef.candidate_passages:
        pq = w2._norm_qty(p.dosage_value, p.dosage_unit) if p.dosage_value is not None else None
        print("  %s single_ok=%s entity_ok=%s dose=%r norm=%r" % (p.id, w2._single_ok(cef, p), w2._entity_ok(cef, p), (p.dosage_value, p.dosage_unit), pq))
    for p, q in itertools.combinations(cef.candidate_passages, 2):
        if (p.id in A and q.id in B) or (p.id in B and q.id in A):
            print("  pair", p.id, q.id, "pair_ok=%s pop=%s role=%s/%s tail=%s/%s" % (
                w2._pair_ok(cef, p, q), w2.pop_compatible(p.population, q.population),
                w2._role(p), w2._role(q), w2._tail(cef, p), w2._tail(cef, q)))
    r = w21.generate_minimal_witness_v21(cef)
    print("v2.1 result: found=%s reason=%s" % (r.found, getattr(r, "reason", None)))
