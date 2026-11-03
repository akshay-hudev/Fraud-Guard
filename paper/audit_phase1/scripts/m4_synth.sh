cd /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged
PY="/Users/apple/Desktop/Akshay@goAI/FraudGuard/.venv-research/bin/python"
M="GB-C GB-CA GB-CAG HAN HGT HGT+AG LR-CA MLP-C Overlap-score RF-CA RGCN SAGE SAGE+AG SimpleHGN"
export RESEARCH_THREADS=2
$PY -m research.experiments.synthetic --shifts 0.0 --densities 0.1 --seeds 11 12 --methods $M --hps "/Users/apple/Desktop/Akshay@goAI/FraudGuard/results/tuning/best_hyperparameters.json" --out /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged/results/synthetic
$PY -m research.experiments.synthetic --shifts 0.0 --densities 0.5 --seeds 7 8 --methods $M --hps "/Users/apple/Desktop/Akshay@goAI/FraudGuard/results/tuning/best_hyperparameters.json" --out /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged/results/synthetic
H="/Users/apple/Desktop/Akshay@goAI/FraudGuard/results/tuning_lowvolume/best_hyperparameters.json"
$PY -m research.experiments.synthetic --shifts 0.0 --densities 0.1 --seeds 1 --ring-claims-per-month 4 --methods $M --hps "$H" --out /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged/results/synthetic_lowvolume_retuned
$PY -m research.experiments.synthetic --shifts 0.0 --densities 0.5 0.9 --seeds 1 2 --ring-claims-per-month 4 --methods $M --hps "$H" --out /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged/results/synthetic_lowvolume_retuned
