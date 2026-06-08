import json
from pathlib import Path

DEFAULT_GOLD_SET_PATH = Path("benchmarks/gold_set.json")


def load_gold_set(path: Path | None = None) -> dict[str, list[str]]:
    """Load gold set mapping each query to a list of relevant OpenAlex IDs."""
    path = path or DEFAULT_GOLD_SET_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("queries", {})
