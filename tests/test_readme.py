"""README.md is generated (scripts/render_readme.py): it must match the artifacts and every link must resolve."""
import re

from ids.data import ROOT
from scripts.render_readme import render

README = (ROOT / "README.md").read_text()


def test_readme_is_up_to_date_with_the_artifacts():
    assert README == render(), "README.md is stale: run .venv/bin/python scripts/render_readme.py"


def test_readme_relative_links_and_anchors_resolve():
    targets = re.findall(r"\]\(([^)]+)\)", README) + re.findall(r'<img src="([^"]+)"', README)
    anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in re.findall(r"^#+ (.+)$", README, re.M)}
    local = [t for t in targets if not t.startswith("http")]
    assert local
    missing = [t for t in local if (t[1:] not in anchors if t.startswith("#") else not (ROOT / t.split("#")[0]).exists())]
    assert missing == []


def test_readme_has_no_machine_paths():
    assert not re.search(r"/Users/|/home/|[A-Z]:\\\\", README)
