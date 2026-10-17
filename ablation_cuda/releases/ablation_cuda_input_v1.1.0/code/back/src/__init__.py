# PDF Graph Analyzer src package
import importlib

def __getattr__(name: str):
    return importlib.import_module(f".{name}", __name__)
