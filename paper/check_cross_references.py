"""Check the numbers the article and its Online Resource use for each other's sections, tables and figures.

The two documents are compiled separately, so they refer to each other through macros (\\STabHGT in the article,
\\MainTabAblation in the Online Resource; \\MainRefNadeau for a reference number of the article). This script
compares every macro with the number LaTeX assigned to the corresponding label or reference, read from the .aux
files of both compiled documents:

    python paper/check_cross_references.py <build dir containing both .tex and .aux files>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# article macro -> label in the Online Resource
TO_SUPPLEMENT = {
    "SSecHyper": "sec:s_hyper", "STabHyper": "tab:hyperparams", "SSecRouting": "sec:s_routing",
    "STabRouting": "tab:cascade", "SFigRouting": "fig:cascade", "SSecDose": "sec:s_dose", "STabDose": "tab:dose",
    "SSecConfirm": "sec:s_confirm", "SSecCost": "sec:s_cost", "SSecHGT": "sec:hgtarch", "STabHGT": "tab:hgt_arch",
    "SSecReliance": "sec:reliance", "SFigAblation": "fig:ablation", "SFigProtocol": "fig:protocolres",
    "SSecKaggle": "sec:s_kaggle", "STabKaggleRepeated": "tab:kaggle_repeated", "STabKaggleNested": "tab:kaggle_nested",
    "SSecLegacy": "sec:legacy", "STabLegacy": "tab:legacy",
}
# Online Resource macro -> label in the article
TO_ARTICLE = {"MainSecLowVolume": "sec:lowvolume", "MainSecMechanism": "sec:mechanism",
              "MainTabAblation": "tab:ablation", "MainTabProtocol": "tab:protocol", "MainTabKaggle": "tab:kaggle",
              "MainTabLowVolume": "tab:lowvolume", "MainRefNadeau": "nadeau2003", "MainRefBouckaert": "bouckaert2004"}


def macros(tex: Path) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex.read_text()))


def labels(aux: Path) -> dict[str, str]:
    """Numbers of labels and of bibliography entries (\\bibcite) in a compiled document."""
    text = aux.read_text()
    return {**dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", text)),
            **dict(re.findall(r"\\bibcite\{([^}]+)\}\{\{([^}]*)\}", text))}


def main() -> None:
    build = Path(sys.argv[1])
    problems = []
    for doc, mapping, other in [("fraud_detection_paper", TO_SUPPLEMENT, "supplementary_material"),
                                ("supplementary_material", TO_ARTICLE, "fraud_detection_paper")]:
        defined, numbered = macros(build / f"{doc}.tex"), labels(build / f"{other}.aux")
        for macro, label in mapping.items():
            if defined.get(macro) != numbered.get(label):
                problems.append(f"{doc}.tex \\{macro} = {defined.get(macro)!r}, but {label} is {numbered.get(label)!r}")
    print("\n".join(problems) if problems else f"all {len(TO_SUPPLEMENT) + len(TO_ARTICLE)} cross-document numbers agree")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
