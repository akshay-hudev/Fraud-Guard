"""Check that a GNN's initial weights depend only on the training seed, not on the world (Online Resource 1, S6).

Tuning and every reported run use training seed 0, so if the initial weights do not depend on the world, each
reported GNN run starts from the weights with which its configuration was selected. Builds SAGE and HGT on a
development world (seed 901) and a test world (seed 3) after seeding with 0 and compares their parameters.

    .venv-research/bin/python paper/hiss_review/check_initial_weights.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.experiments.synthetic import prepare  # noqa: E402
from research.ledger import set_seed  # noqa: E402
from research.models.gnn import GraphModel  # noqa: E402
from research.synth.generator import GeneratorConfig, generate  # noqa: E402

HP = {"hidden": 32, "layers": 3, "heads": 4, "dropout": 0.3, "aggr": "sum"}


def initial_weights(kind: str, graph) -> torch.Tensor:
    set_seed(0)
    model = GraphModel(kind, graph, **HP)
    with torch.no_grad():
        model.eval()
        model(graph)                      # materialise lazily initialised layers
    return torch.cat([p.detach().flatten() for p in model.parameters()])


graphs = []
for seed, shift, density in [(901, 0.5, 0.5), (3, 0.0, 0.1)]:
    train = prepare(generate(GeneratorConfig(seed=seed, ring_shift=shift, ring_density=density)))[2]["train"]
    graphs.append(train[0] if isinstance(train, list) else train)
for kind in ["sage", "hgt"]:
    a, b = (initial_weights(kind, g) for g in graphs)
    print(kind, len(a), "parameters:", "identical" if a.shape == b.shape and torch.equal(a, b) else "different")
