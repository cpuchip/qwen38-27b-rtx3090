"""The frozen document shared by the LABD benchmarks. Stdlib only."""
from pathlib import Path

import harness


def load_corpus(path):
    """Reuse an existing corpus, or freeze this checkout's documentation on first run."""
    path = Path(path)
    repo = Path(harness.REPO)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"{path} is blank; delete it and run bench/labd_bench.py once to rebuild from {repo}")
        return text
    if path.name == "labd_corpus_long.txt":
        raise ValueError(f"{path} is missing; run bench/make_long_corpus.py to build the long corpus")

    files = sorted([*repo.glob("*.md"), *repo.glob("*/README.md")], key=str)
    text = "\n\n".join(p.read_text(encoding="utf-8") for p in files)
    if not text.strip():
        raise ValueError(f"no documentation found in {repo}; cannot build LABD corpus {path}")
    document = text
    while len(text) < 200000:
        text += "\n\n" + document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
