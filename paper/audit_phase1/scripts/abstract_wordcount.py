import re, sys
from pathlib import Path
t = Path("fraud_detection_paper.tex").read_text()
a = t[t.index(r"\abstract{") + 10:t.index(r"\keywords{")]
a = a.rsplit("}", 1)[0]
a = a.replace(r"\mathversion{normal}", " ")
a = re.sub(r"\\textbf\{(.*?)\}", r"\1", a)
a = re.sub(r"\$\\le\$", "≤", a); a = a.replace("\\%", "%")
a = re.sub(r"\s+", " ", a).strip()
print(len(a.split()), "words in the abstract including the four headings;", len(a.split()) - 4, "without them")
