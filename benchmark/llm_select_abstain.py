"""Llama-abstain baseline: same model/settings as llm_select.py, but the prompt allows the answer NONE.
PROMPT IS FROZEN HERE before any run. Run it ONCE on test3. temperature 0, seed 42.
    python benchmark\\llm_select_abstain.py --split benchmark\\data\\test3 --out benchmark\\llama_abstain_test3.json
Also writes <out>.times.json (milliseconds per case) for the runtime table. Resumes if interrupted."""
import argparse, json, os, re, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark.schema import load_split

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="llama3.2:3b"); a = ap.parse_args()
tpath = a.out + ".times.json"
done = json.load(open(a.out)) if os.path.exists(a.out) else {}
times = json.load(open(tpath)) if os.path.exists(tpath) else {}
for c in load_split(a.split):
    if c.case_id in done:
        continue
    lines = "\n".join("[%s] %s" % (p["id"], p["text"]) for p in c.passages)
    prompt = ("Two claims about %s conflict: \"%s\" versus \"%s\".\nBelow are evidence passages. Choose the SMALLEST set of passages "
              "(normally two) that together show this conflict, one passage for each side, with compatible context "
              "(same population, same situation, same kind of quantity).\n"
              "If no pair of passages is a valid, comparable conflict witness, answer NONE.\n"
              "Answer ONLY with the passage ids, comma separated, e.g. p1,p4 -- or the single word NONE.\n\n%s\n\nAnswer:") % (c.entity, c.value_a, c.value_b, lines)
    body = json.dumps({"model": a.model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    txt = json.load(urllib.request.urlopen(req, timeout=600))["response"]
    times[c.case_id] = (time.perf_counter() - t0) * 1000
    ids = [p["id"] for p in c.passages]
    found = [i for i in ids if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(i), txt)]
    done[c.case_id] = [] if (re.search(r"\bNONE\b", txt.upper()) and not found) else found
    json.dump(done, open(a.out, "w"), indent=1); json.dump(times, open(tpath, "w"), indent=1)
    print(c.case_id, "->", done[c.case_id] or "NONE", flush=True)
print("saved", a.out)
