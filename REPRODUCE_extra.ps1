# Extra analyses (run from ., venv active). All label-free of method changes; report as post hoc / upper-bound analyses.
# 1. Oracle threshold (upper bound for the abstaining baselines, threshold chosen on test3 itself)
python benchmark\oracle_tau.py --split benchmark\data\test3 --nli benchmark\nli_test3.json
# 2. Drop-one rule ablation of CWF v2 (identical predictions to v2.1 on test3); method files are not modified
python benchmark\ablate_v2.py --split benchmark\data\test3
# 3. Stronger LLM baseline, ONE run, same frozen prompt (pull the model first: ollama pull llama3.1:8b)
python benchmark\llm_select_abstain.py --split benchmark\data\test3 --model llama3.1:8b --out benchmark\llama8b_abstain_test3.json
# then add it to the stats table with:
python benchmark\stats3.py --split benchmark\data\test3 --nli benchmark\nli_test3.json --llama benchmark\llama_test3.json --llama_abs benchmark\llama8b_abstain_test3.json --ref cwf_v21 --out benchmark\stats_test3_8b.json
