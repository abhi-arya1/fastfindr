# Fast On-Device Vector Search

A high-performance, open-source semantic search system that runs entirely on-device. This project includes a C++ server for processing and a Python Client SDK for easy setup and integration.

## Features

- **On-Device Processing**: All embeddings and search operations run locally
- **Fast Vector Search**: Powered by Faiss for efficient similarity search
- **Semantic Search**: Uses embedding models for semantic understanding
- **REST API**: Simple HTTP API for easy integration
- **Python SDK**: OpenAI-style client library for Python
- **Metadata Filtering**: Filter search results by custom metadata
- **Batch Operations**: Efficient bulk document insertion

## Quick Start

### Server Setup

```bash
cd server
chmod +x setup.sh
./setup.sh # Install Homebrew Dependencies on MacOS 

# Build the server
mkdir -p build && cd build
cmake ..
make server

# Run with default settings
./server

# Or customize configuration
./server --port 8080 --model-path ../embeddinggemma-onnx/model.onnx
```

### Python Client

```python
from client.sdk import Client

# Initialize client
client = Client("http://localhost:8080")

# Create a document
doc = client.documents.create(
    text="Vector databases enable semantic search",
    metadata={"category": "tech"}
)

# Search semantically
results = client.search.semantic("What is similarity search?", k=5)

# Search with metadata filter
results = client.search.by_metadata("category", "tech")
```

## Project Structure

- `/server` - C++ server implementation with vector search engine
- `/client` - Python SDK for interacting with the server
- `/utils` - Utility scripts for model conversion and optimization

## Requirements

### Server
- CMake 3.16+
- C++17 compiler
- Faiss
- ONNX Runtime
- SQLite3
- httplib

### Client
- Python 3.8+
- requests
