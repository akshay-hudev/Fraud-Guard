cd /private/tmp/claude-501/-Users-apple-Desktop-Akshay-goAI-FraudGuard/243a9ecf-d62f-4d66-ba55-bb3a5a3cca8e/scratchpad/tagged
PY="/Users/apple/Desktop/Akshay@goAI/FraudGuard/.venv-research/bin/python"
export RESEARCH_THREADS=2
$PY -m research.experiments.kaggle_exp --stage nested --nested-gnns --methods SAGE SAGE+G --tag gnn_sage
$PY -m research.experiments.kaggle_exp --stage nested --nested-gnns --methods HAN --tag gnn_han
$PY -m research.experiments.kaggle_exp --stage nested --nested-gnns --methods RGCN --tag gnn_rgcn
