#pragma once

#include <vector>
#include <string>
#include <memory>
#include <faiss/IndexHNSW.h>
#include "inference.h"
#include "storage.h"

struct SearchResult {
    std::string text;
    float score;
    size_t id;
    std::map<std::string, std::string> metadata;
};

class VectorSearch {
public:
    VectorSearch(const std::string& modelPath, const std::string& tokenizerPath, 
                 const std::string& dbPath, int M = 16, int efConstruction = 200);
    ~VectorSearch();
    
    bool initialize();
    void loadOrCreateIndex(const std::string& index_file);
    
    size_t addDocument(const std::string& text, const std::map<std::string, std::string>& metadata = {});
    void addDocuments(const std::vector<std::string>& texts, 
                      const std::vector<std::map<std::string, std::string>>& metadataList = {});
    bool updateDocument(size_t id, const std::string& text, const std::map<std::string, std::string>& metadata = {});
    bool deleteDocument(size_t id);
    
    std::vector<SearchResult> searchText(const std::string& query, int k = 10);
    std::vector<SearchResult> searchEmbedding(const std::vector<float>& queryEmbedding, int k = 10);
    std::vector<SearchResult> searchByMetadata(const std::string& key, const std::string& value, int k = 10);
    
    void saveIndex(const std::string& index_file);
    void rebuildIndex();
    
    Document getDocument(size_t id);
    std::vector<Document> getAllDocuments();
    size_t getDocumentCount();
    
    bool isInitialized() const { return index != nullptr && storage_ && storage_->isOpen(); }
    bool isModelLoaded() const { return inferenceEngine_ && inferenceEngine_->isLoaded(); }
    long getIndexSize() const { return index ? index->ntotal : 0; }
    size_t getEmbeddingDimension() const;
    Storage* getStorage() const { return storage_.get(); }

private:
    std::unique_ptr<InferenceEngine> inferenceEngine_;
    std::unique_ptr<Storage> storage_;
    std::string modelPath_;
    std::string tokenizerPath_;
    std::string dbPath_;
    bool useCuda_;
    
    int d;  // embedding dimension
    faiss::IndexHNSWFlat* index;
    std::vector<size_t> indexToDocumentId_;  // Maps FAISS index positions to document IDs
    
    void initializeIndex();
    std::vector<float> getEmbedding(const std::string& text);
    void synchronizeIndex();
    SearchResult buildSearchResult(size_t documentId, float score);
};