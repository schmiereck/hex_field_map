"""Project directory layout: figures/, data/ and docs/ next to src/."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DIR = os.path.join(ROOT, "docs")


def fig_path(name):
    """Path for a generated figure/animation."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def data_path(name):
    """Path for a numerical result file (.npz)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def docs_path(name):
    """Path for a generated markdown report."""
    return os.path.join(DOCS_DIR, name)
