# Run from . with the venv active. Order matters: baselines are fixed BEFORE the single test3 run.
# 0. verify nothing frozen changed
python benchmark\freeze.py verify benchmark\data\test3
# 1. Llama-abstain: ONE run on test3 (prompt frozen inside the script). Needs Ollama running with llama3.2:3b.
python benchmark\llm_select_abstain.py --split benchmark\data\test3 --out benchmark\llama_abstain_test3.json
# 2. all metrics + statistics (Wilson, McNemar+Holm, bootstrap, abstention P/R/F1, runtime)
python benchmark\stats3.py --split benchmark\data\test3 --nli benchmark\nli_test3.json --llama benchmark\llama_test3.json --llama_abs benchmark\llama_abstain_test3.json --ref cwf_v21 --out benchmark\stats_test3.json
# 3. selective-prediction figure
python benchmark\selective3.py --split benchmark\data\test3 --nli benchmark\nli_test3.json --out benchmark\selective_test3
# 4. classmate agreement (after answers_C.txt is filled)
python benchmark\agree_C.py --c benchmark\answers_C.txt --split benchmark\data\test3 --a benchmark\annotations_A3.json --b benchmark\annotations_B3.json --out benchmark\agreement_C.txt
