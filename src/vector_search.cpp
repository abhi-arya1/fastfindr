#include "vector_search.h"
#include <iostream>
#include <random>
#include <fstream>
#include <filesystem>
#include <faiss/index_io.h>

VectorSearch::VectorSearch(int dimension, int M, int efConstruction) 
    : d(dimension), index(nullptr), database_vectors(nullptr), nb(0) {
    index = new faiss::IndexHNSWFlat(d, M);
    index->hnsw.efConstruction = efConstruction;
}

VectorSearch::~VectorSearch() {
    delete index;
    delete[] database_vectors;
}

void VectorSearch::loadOrCreateIndex(const std::string& index_file, const std::string& vectors_file, int database_size) {
    nb = database_size;
    database_vectors = new float[d * nb];
    
    if (std::filesystem::exists(index_file) && std::filesystem::exists(vectors_file)) {
        printf("Loading existing index and vectors...\n");
        
        // Load index
        delete index;  // Clean up existing index
        index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(index_file.c_str()));
        
        // Load vectors
        loadVectors(vectors_file);
        
        printf("Loaded HNSW index with %lld vectors\n", index->ntotal);
    } else {
        printf("Creating new HNSW index and vectors...\n");
        
        // Generate random database vectors
        generateRandomVectors(database_vectors, nb, d);
        
        printf("is_trained = %s\n", index->is_trained ? "true" : "false");
        index->add(nb, database_vectors);
        printf("ntotal = %lld\n", index->ntotal);
        
        // Save index and vectors
        saveIndex(index_file);
        saveVectors(vectors_file);
        
        printf("Saved HNSW index and vectors to disk\n");
    }
}

void VectorSearch::search(const float* query_vectors, int nq, int k, float* distances, faiss::idx_t* indices) {
    if (!index) {
        printf("Error: Index not initialized\n");
        return;
    }
    
    index->search(nq, query_vectors, k, distances, indices);
}

void VectorSearch::saveIndex(const std::string& index_file) {
    if (index) {
        faiss::write_index(index, index_file.c_str());
    }
}

void VectorSearch::generateRandomVectors(float* vectors, int count, int dimension) {
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    
    for(int i = 0; i < dimension * count; i++) {
        vectors[i] = distrib(rng);
    }
}

void VectorSearch::loadVectors(const std::string& vectors_file) {
    std::ifstream file(vectors_file, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(database_vectors), d * nb * sizeof(float));
        file.close();
    }
}

void VectorSearch::saveVectors(const std::string& vectors_file) {
    std::ofstream file(vectors_file, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(database_vectors), d * nb * sizeof(float));
        file.close();
    }
}