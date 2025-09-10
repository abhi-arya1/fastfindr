import sys
from pathlib import Path
from time import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import Client 
client = Client()

COUNTRIES = Path(__file__).parent / "data/countries"

texts = []
start_time = time()

for file in COUNTRIES.iterdir():
    if file.suffix == ".txt":
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())

end_time = time()
print(f"Loaded {len(texts)} documents in {end_time - start_time:.2f} seconds.")

for text in texts:
    client.documents.create(text=text)

print(f"Indexed {len(list(COUNTRIES.iterdir()))} documents.")

start_time = time()
results = client.search.query("population of portugal", threshold=0.7)
end_time = time()
print(f"Search completed in {end_time - start_time:.2f} seconds.")

for result in results:
    print("ID:", result["id"])
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['text'][:200]}...")
    print()
