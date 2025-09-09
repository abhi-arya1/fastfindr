from subprocess import run 
from pathlib import Path

MODEL = "google/embeddinggemma-300m"

OUTPUT = Path(__file__).parent.parent / "embeddinggemma-onnx"

print(f"Starting export of {MODEL} to {OUTPUT}")

run([
    "optimum-cli",
    "export",
    "--model",
    MODEL,
    OUTPUT
])

print(f"Model {MODEL} exported to {OUTPUT}")
