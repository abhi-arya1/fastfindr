#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include "tokenizers_cpp.h"

constexpr size_t DEFAULT_EMBEDDING_DIMENSION = 768;

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    bool loadModel(const std::string& modelPath, const std::string& tokenizerPath, bool useCuda = false);
    void unloadModel();
    bool isLoaded() const;
    
    std::vector<float> getEmbedding(const std::string& text, int64_t maxLen = 256);
    std::vector<std::vector<float>> getEmbeddings(const std::vector<std::string>& texts, int64_t maxLen = 256);
    std::vector<float> cosineSimMatrix(const std::vector<std::vector<float>>& embeddings);
    
    size_t getEmbeddingDimension() const { return embeddingDim_; }
    
private:
    struct Batch {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;
        int64_t B{0}, S{0};
    };
    
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    Ort::MemoryInfo memoryInfo_;
    
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamesCStr_;
    std::vector<const char*> outputNamesCStr_;
    
    size_t embeddingDim_;
    bool loaded_;
    
    void initializeSession(const std::string& modelPath, bool useCuda);
    void loadTokenizer(const std::string& tokenizerPath);
    void extractModelInfo();
    
    Batch tokenizeBatch(const std::vector<std::string>& texts, int64_t maxLen);
    std::vector<Ort::Value> createInputTensors(const Batch& batch);
    std::vector<float> meanPoolL2Norm(const float* lastHidden, const int64_t* mask, 
                                     int64_t B, int64_t S, int64_t H);
    
    std::vector<uint8_t> readFileBytes(const std::string& path);
    
#ifdef _WIN32
    std::wstring toWideString(const std::string& s);
#endif
};
