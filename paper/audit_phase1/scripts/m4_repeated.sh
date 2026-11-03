cd /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged
PY="/Users/apple/Desktop/Akshay@goAI/FraudGuard/.venv-research/bin/python"
M="LR-P RF-P GB-P GB-PG GB-PGL MLP-P SAGE HAN HGT"
export RESEARCH_THREADS=2
$PY -m research.experiments.kaggle_exp --stage eval --repeat 3 --folds 1 2 3 4 --methods $M --tag main
$PY -m research.experiments.kaggle_exp --stage eval --repeat 4 --folds 0 1 2 3 4 --methods $M --tag main
