from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# Initialize the model
model = SentenceTransformer('google/embeddinggemma-300m')

# Directory containing country files
countries_dir = Path(__file__).parent.parent / 'examples' / 'data' / 'countries'

# Read all country files
country_texts = {}
for filename in os.listdir(countries_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(countries_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            country_name = filename.replace('.txt', '').capitalize()
            country_texts[country_name] = content

# Search queries
queries = ["Landlocked Country", "Portugal", "East Asia", "South Asia", "Europe", "Eurasia", "Americas"]

# Encode all country texts
country_names = list(country_texts.keys())
country_contents = list(country_texts.values())
corpus_embeddings = model.encode(country_contents)

# Search for each query
for query in queries:
    print(f"\nSearch query: '{query}'")
    print("-" * 40)
    
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate similarities
    from sentence_transformers import util
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Get top result
    top_result_idx = similarities.argmax()
    top_country = country_names[top_result_idx]
    top_score = similarities[top_result_idx].item()
    
    print(f"Top result: {top_country} (score: {top_score:.4f})")
    print(f"Preview: {country_texts[top_country][:200]}...")
