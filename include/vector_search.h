#pragma once

#include <vector>
#include <string>
#include <faiss/IndexHNSW.h>

class VectorSearch {
public:
    VectorSearch(int dimension, int M = 16, int efConstruction = 200);
    ~VectorSearch();
    
    void loadOrCreateIndex(const std::string& index_file, const std::string& vectors_file, int nb);
    void search(const float* query_vectors, int nq, int k, float* distances, faiss::idx_t* indices);
    void saveIndex(const std::string& index_file);
    
    bool isIndexLoaded() const { return index != nullptr; }
    long getIndexSize() const { return index ? index->ntotal : 0; }

private:
    int d;  // dimension
    faiss::IndexHNSWFlat* index;
    float* database_vectors;
    int nb;  // number of database vectors
    
    void generateRandomVectors(float* vectors, int count, int dimension);
    void loadVectors(const std::string& vectors_file);
    void saveVectors(const std::string& vectors_file);
};