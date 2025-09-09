#include <iostream>
#include <vector>
#include <string>
#include "vector_search.h"
#include "embedding_loader.h"
#include "tokenizer.h"

int main() {
    // Four sentences to load into database
    std::vector<std::string> sentences = {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning algorithms process vast amounts of data", 
        "The ocean waves crashed against the rocky shore",
        "Artificial intelligence will revolutionize modern computing"
    };
    
    std::cout << "Initializing tokenizer and embedding model..." << std::endl;
    HFTokenizer tokenizer;
    EmbeddingLoader embeddingLoader;
    std::string modelPath = "model.onnx";
    
    if (!embeddingLoader.loadModel(modelPath)) {
        std::cout << "Failed to load ONNX model. Download from:" << std::endl;
        std::cout << "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX" << std::endl;
        return 1;
    }
    
    int d = embeddingLoader.getEmbeddingDimension();
    std::cout << "Embedding dimension: " << d << std::endl;
    
    std::cout << "\nGenerating embeddings for sentences..." << std::endl;
    std::vector<std::vector<float>> sentence_embeddings;
    
    for (const auto& sentence : sentences) {
        auto tokens = tokenizer.encode(sentence);
        
        try {
            auto embedding = embeddingLoader.getEmbedding(tokens);
            sentence_embeddings.push_back(embedding);
            std::cout << "Generated embedding for: \"" << sentence << "\"" << std::endl;
            std::cout << "  Tokens: " << tokens.size() << " | First few: ";
            for (size_t i = 0; i < std::min(tokens.size(), size_t(5)); i++) {
                std::cout << tokens[i] << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error generating embedding: " << e.what() << std::endl;
        }
    }
    
    if (sentence_embeddings.empty()) {
        std::cout << "No embeddings generated!" << std::endl;
        return 1;
    }
    
    std::cout << "\nBuilding search index..." << std::endl;
    
    // Convert embeddings to float array for FAISS
    int nb = sentence_embeddings.size();
    float* database_vectors = new float[d * nb];
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            database_vectors[i * d + j] = sentence_embeddings[i][j];
        }
    }
    
    // Add embeddings to search index
    faiss::IndexHNSWFlat* index = new faiss::IndexHNSWFlat(d, 16);
    index->hnsw.efConstruction = 200;
    index->add(nb, database_vectors);
    
    std::cout << "Index built with " << index->ntotal << " sentences" << std::endl;
    
    // Search for a target word
    std::string target_word = "ocean";
    std::cout << "\nSearching for word: \"" << target_word << "\"" << std::endl;
    
    // Generate embedding for the target word using tokenizer
    auto target_tokens = tokenizer.encode(target_word);
    std::cout << "Target word tokens: ";
    for (auto token : target_tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    try {
        auto query_embedding = embeddingLoader.getEmbedding(target_tokens);
        
        // Search for nearest sentence
        int k = 1;
        faiss::idx_t* I = new faiss::idx_t[k];
        float* D = new float[k];
        
        index->search(1, query_embedding.data(), k, D, I);
        
        std::cout << "\nFound match:" << std::endl;
        std::cout << "Sentence " << I[0] << ": \"" << sentences[I[0]] << "\"" << std::endl;
        std::cout << "Distance: " << D[0] << std::endl;
        
        delete[] I;
        delete[] D;
    } catch (const std::exception& e) {
        std::cout << "Error searching: " << e.what() << std::endl;
    }
    
    delete[] database_vectors;
    delete index;
    embeddingLoader.unloadModel();
    
    return 0;
}