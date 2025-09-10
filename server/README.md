# Server - High-Performance Vector Search Engine

## Building

### Prerequisites

Install dependencies (macOS with Homebrew):

```bash
./setup.sh
```

### Build Instructions

```bash
cd server
mkdir -p build && cd build
cmake ..
make server
```

## Running the Server

### Basic Usage

```bash
# Run with defaults (port 8080)
./server

# Custom configuration
./server --port 9000 --database-path my_data.db --index-path my_vectors.index

# With custom model
./server --model-path /path/to/model.onnx --tokenizer-path /path/to/tokenizer.json

# Create new database (clears existing)
./server --create-new-db

# Verbose logging
./server --log-level verbose
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | localhost | Server host address |
| `--port` | 8080 | Server port |
| `--model-path` | ../embeddinggemma-onnx/model.onnx | Path to ONNX model |
| `--tokenizer-path` | ../embeddinggemma-onnx/tokenizer.json | Path to tokenizer |
| `--database-path` | database.db | SQLite database path |
| `--index-path` | vectors.index | Faiss index path |
| `--create-new-db` | false | Create fresh database |
| `--log-level` | info | Logging level (verbose/info/warning/error) |

## API Reference

### Health Check

```http
GET /health
```

Returns server status and statistics.

### Document Operations

#### Create Document
```http
POST /documents
Content-Type: application/json

{
  "text": "Document text content",
  "metadata": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

#### Get Document
```http
GET /documents/{id}
```

#### Update Document
```http
PUT /documents/{id}
Content-Type: application/json

{
  "text": "Updated text",
  "metadata": {
    "key": "value"
  }
}
```

#### Delete Document
```http
DELETE /documents/{id}
```

#### List Documents
```http
GET /documents?key=category&value=tech
```

#### Batch Insert
```http
POST /documents/batch
Content-Type: application/json

{
  "documents": [
    {
      "text": "First document",
      "metadata": {"type": "article"}
    },
    {
      "text": "Second document",
      "metadata": {"type": "blog"}
    }
  ]
}
```

### Search Operations

#### Semantic Search
```http
POST /search
Content-Type: application/json

{
  "query": "search query text",
  "k": 10,
  "type": "semantic",
  "metadata": {
    "key": "category",
    "value": "tech"
  }
}
```

Search types:
- `semantic` - Vector similarity search (default)
- `text` - Full-text search using SQL LIKE
- `fulltext` - Alias for text search

### Index Management

#### Rebuild Index
```http
POST /index/rebuild
```

#### Save Index
```http
POST /index/save
```

## Architecture

### Components

- **SearchServer**: Main HTTP server class handling routes
- **VectorSearch**: Core search engine with embedding and indexing
- **Storage**: SQLite interface for document persistence
- **Inference**: ONNX Runtime wrapper for embeddings

### File Structure

```
server/
├── include/         # Header files
│   ├── server.h
│   ├── vector_search.h
│   ├── storage.h
│   └── inference.h
├── src/            # Implementation files
│   ├── server.cpp
│   ├── vector_search.cpp
│   ├── storage.cpp
│   └── inference.cpp
├── third_party/    # Dependencies
│   └── tokenizers-cpp/
├── embeddinggemma-onnx/  # Model files
│   ├── model.onnx
│   └── tokenizer.json
└── CMakeLists.txt
```
<!-- 
## Performance Tuning

### Memory Usage
- Adjust batch size for embedding generation
- Configure Faiss index parameters for memory/speed tradeoff

### Concurrency
- OpenMP automatically uses available CPU cores
- Set `OMP_NUM_THREADS` environment variable to control parallelism

### Index Optimization
- Rebuild index periodically for optimal performance
- Save index to disk to avoid rebuilding on restart

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Verify ONNX model path is correct
   - Ensure model is compatible with ONNX Runtime version

2. **Build errors**
   - Check all dependencies are installed
   - Verify CMake can find libraries (check CMakeLists.txt paths)

3. **Performance issues**
   - Enable OpenMP support
   - Use optimized ONNX models
   - Consider index parameters for your dataset size

## Development

### Adding New Endpoints

1. Add handler method to `SearchServer` class in `server.h`
2. Implement handler in `server.cpp`
3. Register route in `setupRoutes()` method

### Testing

```bash
# Health check
curl http://localhost:8080/health

# Create document
curl -X POST http://localhost:8080/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "Test document"}'

# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'
``` -->

This is licensed under [LICENSE](../LICENSE).