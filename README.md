# FAISS C++ Example

A simple demonstration of the FAISS (Facebook AI Similarity Search) library in C++.

## Build

```bash
mkdir build
cd build
cmake ..
make
./faiss_example
```

## Overview

Creates 100,000 random 64-dimensional vectors, indexes them with FAISS, then searches for the 4 nearest neighbors of 10,000 query vectors using L2 distance.

## Output

Shows indices and distances of nearest neighbors for the first 5 queries:
- **I=**: Database vector indices closest to each query
- **D=**: L2 distances (smaller = more similar)