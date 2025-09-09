#pragma once

#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

class EmbeddingLoader {
public:
    EmbeddingLoader();
    ~EmbeddingLoader();
    
    bool loadModel(const std::string& modelPath);
    void unloadModel();
    bool isLoaded() const;
    
    std::vector<float> getEmbedding(const std::vector<int64_t>& inputIds);
    std::vector<std::vector<float>> getEmbeddings(const std::vector<std::vector<int64_t>>& batchInputIds);
    
    size_t getEmbeddingDimension() const { return embeddingDim_; }
    
private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memoryInfo_;
    
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::vector<std::vector<int64_t>> outputShapes_;
    
    size_t embeddingDim_;
    bool loaded_;
    
    void initializeSession(const std::string& modelPath);
    void extractModelInfo();
};