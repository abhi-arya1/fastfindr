from subprocess import run 
from pathlib import Path

MODEL = "google/embeddinggemma-300m"

OUTPUT = Path(__file__).parent.parent / "embeddinggemma-onnx"

print(f"Starting export of {MODEL} to {OUTPUT}")

run([
    str(Path(__file__).parent / "venv" / "bin" / "optimum-cli"),
    "export",
    "onnx",
    "--model",
    MODEL,
    str(OUTPUT),
    # "--quantize"
])

print(f"Model {MODEL} exported to {OUTPUT}")
