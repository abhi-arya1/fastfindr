#include "vector_search.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <faiss/index_io.h>
#include <algorithm>

VectorSearch::VectorSearch(const std::string& modelPath, const std::string& tokenizerPath, 
                         const std::string& dbPath, int M, int efConstruction)
    : modelPath_(modelPath), tokenizerPath_(tokenizerPath), dbPath_(dbPath)
    , d(0), index(nullptr) {
    inferenceEngine_ = std::make_unique<InferenceEngine>();
    storage_ = std::make_unique<Storage>(dbPath_);
}

VectorSearch::~VectorSearch() {
    delete index;
}

bool VectorSearch::initialize() {
    if (!storage_->initialize()) {
        std::cerr << "Failed to initialize storage" << std::endl;
        return false;
    }
    
    if (!inferenceEngine_->loadModel(modelPath_, tokenizerPath_, useCuda_)) {
        std::cerr << "Failed to load inference model" << std::endl;
        return false;
    }
    
    d = static_cast<int>(inferenceEngine_->getEmbeddingDimension());
    std::cout << "Model loaded with embedding dimension: " << d << std::endl;
    
    return true;
}

void VectorSearch::loadOrCreateIndex(const std::string& index_file) {
    if (!isModelLoaded()) {
        std::cerr << "Error: Model not loaded. Call initialize() first." << std::endl;
        return;
    }
    
    if (std::filesystem::exists(index_file)) {
        printf("Loading existing index...\n");
        
        delete index;
        index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(index_file.c_str()));
        
        synchronizeIndex();
        
        printf("Loaded HNSW index with %lld vectors\n", index->ntotal);
    } else {
        printf("Creating new HNSW index...\n");
        initializeIndex();
        rebuildIndex();
        printf("Created and populated HNSW index\n");
    }
}

std::vector<SearchResult> VectorSearch::searchText(const std::string& query, int k, float threshold, int efSearch) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return {};
    }
    
    auto queryEmbedding = getEmbedding(query);
    return searchEmbedding(queryEmbedding, k, threshold, efSearch);
}

std::vector<SearchResult> VectorSearch::searchEmbedding(const std::vector<float>& queryEmbedding, int k, float threshold, int efSearch) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return {};
    }
    
    if (index->ntotal == 0) {
        std::cerr << "Error: Index is empty" << std::endl;
        return {};
    }
    
    k = std::min(k, static_cast<int>(index->ntotal));
    
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);
    
    index->hnsw.efSearch = efSearch;
    index->search(1, queryEmbedding.data(), k, distances.data(), indices.data());
    
    std::vector<SearchResult> results;
    results.reserve(k);
    
    for (int i = 0; i < k; ++i) {
        if (indices[i] >= 0 && indices[i] < static_cast<faiss::idx_t>(indexToDocumentId_.size())) {
            size_t documentId = indexToDocumentId_[indices[i]];
            float score = 1.0f / (1.0f + distances[i]); // Convert distance to similarity score
            
            // Only include results above the threshold
            if (score >= threshold) {
                results.push_back(buildSearchResult(documentId, score));
            }
        }
    }
    
    return results;
}

std::vector<SearchResult> VectorSearch::searchByMetadata(const std::string& key, const std::string& value, int k) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return {};
    }
    
    auto documents = storage_->getDocumentsByMetadata(key, value);
    std::vector<SearchResult> results;
    
    int count = std::min(k, static_cast<int>(documents.size()));
    for (int i = 0; i < count; ++i) {
        SearchResult result;
        result.id = documents[i].id;
        result.text = documents[i].text;
        result.metadata = documents[i].metadata;
        result.score = 1.0f; // Exact metadata match
        results.push_back(result);
    }
    
    return results;
}

void VectorSearch::saveIndex(const std::string& index_file) {
    if (index) {
        faiss::write_index(index, index_file.c_str());
    }
}

size_t VectorSearch::addDocument(const std::string& text, const std::map<std::string, std::string>& metadata) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return 0;
    }
    
    // Add document to storage first
    size_t documentId = storage_->addDocument(text, metadata);
    if (documentId == 0) {
        std::cerr << "Failed to add document to storage" << std::endl;
        return 0;
    }
    
    // Generate embedding
    auto embedding = getEmbedding(text);
    if (embedding.empty()) {
        std::cerr << "Failed to generate embedding" << std::endl;
        storage_->deleteDocument(documentId);
        return 0;
    }
    
    if (!index) {
        initializeIndex();
    }
    
    // Add to FAISS index
    index->add(1, embedding.data());
    indexToDocumentId_.push_back(documentId);
    
    std::cout << "Added document " << documentId << " to index. Total: " << index->ntotal << std::endl;
    return documentId;
}

void VectorSearch::addDocuments(const std::vector<std::string>& texts, 
                               const std::vector<std::map<std::string, std::string>>& metadataList) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return;
    }
    
    storage_->beginTransaction();
    
    try {
        std::vector<size_t> documentIds;
        documentIds.reserve(texts.size());
        
        // Add documents to storage
        for (size_t i = 0; i < texts.size(); ++i) {
            const auto& metadata = (i < metadataList.size()) ? metadataList[i] : std::map<std::string, std::string>{};
            size_t documentId = storage_->addDocument(texts[i], metadata);
            if (documentId == 0) {
                throw std::runtime_error("Failed to add document to storage");
            }
            documentIds.push_back(documentId);
        }
        
        // Generate embeddings
        auto embeddings = inferenceEngine_->getEmbeddings(texts);
        if (embeddings.size() != texts.size()) {
            throw std::runtime_error("Embedding generation failed");
        }
        
        // Add to FAISS index
        std::vector<float> flatEmbeddings;
        flatEmbeddings.reserve(embeddings.size() * d);
        
        for (const auto& emb : embeddings) {
            flatEmbeddings.insert(flatEmbeddings.end(), emb.begin(), emb.end());
        }
        
        if (!index) {
            initializeIndex();
        }
        
        index->add(static_cast<int>(embeddings.size()), flatEmbeddings.data());
        indexToDocumentId_.insert(indexToDocumentId_.end(), documentIds.begin(), documentIds.end());
        
        storage_->commitTransaction();
        
        std::cout << "Added " << texts.size() << " documents to index. Total: " << index->ntotal << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error adding documents: " << e.what() << std::endl;
        storage_->rollbackTransaction();
    }
}

bool VectorSearch::updateDocument(size_t id, const std::string& text, const std::map<std::string, std::string>& metadata) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return false;
    }
    
    if (!storage_->updateDocument(id, text, metadata)) {
        return false;
    }
    
    // Rebuild index to reflect changes
    rebuildIndex();
    return true;
}

bool VectorSearch::deleteDocument(size_t id) {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return false;
    }
    
    if (!storage_->deleteDocument(id)) {
        return false;
    }
    
    // Rebuild index to reflect changes
    rebuildIndex();
    return true;
}

void VectorSearch::rebuildIndex() {
    if (!isInitialized()) {
        std::cerr << "Error: System not initialized" << std::endl;
        return;
    }
    
    std::cout << "Rebuilding index..." << std::endl;
    
    // Clear existing index
    delete index;
    initializeIndex();
    indexToDocumentId_.clear();
    
    // Get all documents
    auto documents = storage_->getAllDocuments();
    if (documents.empty()) {
        std::cout << "No documents to index" << std::endl;
        return;
    }
    
    // Extract texts and generate embeddings
    std::vector<std::string> texts;
    texts.reserve(documents.size());
    
    for (const auto& doc : documents) {
        texts.push_back(doc.text);
    }
    
    auto embeddings = inferenceEngine_->getEmbeddings(texts);
    
    // Flatten embeddings and add to index
    std::vector<float> flatEmbeddings;
    flatEmbeddings.reserve(embeddings.size() * d);
    
    for (const auto& emb : embeddings) {
        flatEmbeddings.insert(flatEmbeddings.end(), emb.begin(), emb.end());
    }
    
    index->add(static_cast<int>(embeddings.size()), flatEmbeddings.data());
    
    // Update mapping
    for (const auto& doc : documents) {
        indexToDocumentId_.push_back(doc.id);
    }
    
    std::cout << "Rebuilt index with " << index->ntotal << " vectors" << std::endl;
}

void VectorSearch::synchronizeIndex() {
    if (!storage_ || !storage_->isOpen()) {
        std::cerr << "Error: Storage not available" << std::endl;
        return;
    }
    
    // Get all document IDs from storage
    auto allDocumentIds = storage_->getAllDocumentIds();
    indexToDocumentId_ = allDocumentIds;
    
    if (index && index->ntotal != static_cast<long>(allDocumentIds.size())) {
        std::cout << "Index size mismatch. Rebuilding..." << std::endl;
        rebuildIndex();
    }
}

Document VectorSearch::getDocument(size_t id) {
    if (!storage_) {
        return Document();
    }
    return storage_->getDocument(id);
}

std::vector<Document> VectorSearch::getAllDocuments() {
    if (!storage_) {
        return {};
    }
    return storage_->getAllDocuments();
}

size_t VectorSearch::getDocumentCount() {
    if (!storage_) {
        return 0;
    }
    return storage_->getDocumentCount();
}

size_t VectorSearch::getEmbeddingDimension() const {
    return inferenceEngine_ ? inferenceEngine_->getEmbeddingDimension() : 0;
}

void VectorSearch::initializeIndex() {
    if (d <= 0) {
        std::cerr << "Error: Invalid embedding dimension" << std::endl;
        return;
    }
    
    index = new faiss::IndexHNSWFlat(d, 32); // M = 16
    index->hnsw.efConstruction = 400;
    
    std::cout << "Initialized HNSW index with dimension " << d << std::endl;
}

std::vector<float> VectorSearch::getEmbedding(const std::string& text) {
    if (!isModelLoaded()) {
        std::cerr << "Error: Model not loaded" << std::endl;
        return {};
    }
    
    return inferenceEngine_->getEmbedding(text);
}

SearchResult VectorSearch::buildSearchResult(size_t documentId, float score) {
    Document doc = storage_->getDocument(documentId);
    
    SearchResult result;
    result.id = doc.id;
    result.text = doc.text;
    result.metadata = doc.metadata;
    result.score = score;
    
    return result;
}