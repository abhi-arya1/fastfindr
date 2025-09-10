#include "inference.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

InferenceEngine::InferenceEngine() 
    : memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , embeddingDim_(DEFAULT_EMBEDDING_DIMENSION)
    , loaded_(false) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine");
}

InferenceEngine::~InferenceEngine() {
    unloadModel();
}

bool InferenceEngine::loadModel(const std::string& modelPath, const std::string& tokenizerPath, bool useCuda) {
    try {
        loadTokenizer(tokenizerPath);
        initializeSession(modelPath, useCuda);
        extractModelInfo();
        loaded_ = true;
        std::cout << "Model loaded successfully. Embedding dimension: " << embeddingDim_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

void InferenceEngine::unloadModel() {
    if (session_) {
        session_.reset();
        tokenizer_.reset();
        inputNames_.clear();
        outputNames_.clear();
        inputNamesCStr_.clear();
        outputNamesCStr_.clear();
        embeddingDim_ = 0;
        loaded_ = false;
        std::cout << "Model unloaded." << std::endl;
    }
}

bool InferenceEngine::isLoaded() const {
    return loaded_;
}

std::vector<float> InferenceEngine::getEmbedding(const std::string& text, int64_t maxLen) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    std::vector<std::string> texts = {text};
    auto embeddings = getEmbeddings(texts, maxLen);
    return embeddings[0];
}

std::vector<std::vector<float>> InferenceEngine::getEmbeddings(const std::vector<std::string>& texts, int64_t maxLen) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    Batch batch = tokenizeBatch(texts, maxLen);
    std::vector<Ort::Value> ortInputs = createInputTensors(batch);
    
    auto ortOutputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNamesCStr_.data(),
        ortInputs.data(),
        ortInputs.size(),
        outputNamesCStr_.data(),
        outputNamesCStr_.size()
    );
    
    auto& lastHiddenVal = ortOutputs.front();
    auto shapeInfo = lastHiddenVal.GetTensorTypeAndShapeInfo();
    auto dims = shapeInfo.GetShape();
    
    if (dims.size() != 3) {
        throw std::runtime_error("Unexpected output rank; expected 3.");
    }
    
    int64_t B = dims[0];
    int64_t S = dims[1]; 
    int64_t H = dims[2];
    
    const float* lastHidden = lastHiddenVal.GetTensorData<float>();
    std::vector<float> flatEmbeddings = meanPoolL2Norm(lastHidden, batch.attention_mask.data(), B, S, H);
    
    std::vector<std::vector<float>> result;
    result.reserve(B);
    
    for (int64_t i = 0; i < B; ++i) {
        std::vector<float> embedding(flatEmbeddings.begin() + i * H, flatEmbeddings.begin() + (i + 1) * H);
        result.push_back(embedding);
    }
    
    return result;
}

std::vector<float> InferenceEngine::cosineSimMatrix(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        return {};
    }
    
    int64_t B = embeddings.size();
    int64_t H = embeddings[0].size();
    
    std::vector<float> flatEmbeddings;
    flatEmbeddings.reserve(B * H);
    
    for (const auto& emb : embeddings) {
        flatEmbeddings.insert(flatEmbeddings.end(), emb.begin(), emb.end());
    }
    
    std::vector<float> M(B * B, 0.f);
    for (int64_t i = 0; i < B; ++i) {
        const float* ei = &flatEmbeddings[i * H];
        for (int64_t j = 0; j < B; ++j) {
            const float* ej = &flatEmbeddings[j * H];
            float dot = 0.f;
            for (int64_t h = 0; h < H; ++h) {
                dot += ei[h] * ej[h];
            }
            M[i * B + j] = dot;
        }
    }
    
    return M;
}

void InferenceEngine::initializeSession(const std::string& modelPath, bool useCuda) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (useCuda) {
        //
        // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
        std::cout << "Warning: CUDA requested but not available on this platform" << std::endl;
    }
    
#ifdef _WIN32
    session_ = std::make_unique<Ort::Session>(*env_, toWideString(modelPath).c_str(), sessionOptions);
#else
    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif
}

void InferenceEngine::loadTokenizer(const std::string& tokenizerPath) {
    auto blob = readFileBytes(tokenizerPath);
    std::string json(reinterpret_cast<const char*>(blob.data()), blob.size());
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json);
}

void InferenceEngine::extractModelInfo() {
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t numInputNodes = session_->GetInputCount();
    inputNames_.reserve(numInputNodes);
    inputNamesCStr_.reserve(numInputNodes);
    
    for (size_t i = 0; i < numInputNodes; i++) {
        auto inputName = session_->GetInputNameAllocated(i, allocator);
        inputNames_.push_back(inputName.get());
    }
    
    for (const auto& name : inputNames_) {
        inputNamesCStr_.push_back(name.c_str());
    }
    
    size_t numOutputNodes = session_->GetOutputCount();
    outputNames_.reserve(numOutputNodes);
    outputNamesCStr_.reserve(numOutputNodes);
    
    for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.push_back(outputName.get());
        
        if (i == 0) {
            Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(i);
            auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            auto outputShape = outputTensorInfo.GetShape();
            
            if (!outputShape.empty()) {
                embeddingDim_ = static_cast<size_t>(outputShape.back());
            }
        }
    }
    
    for (const auto& name : outputNames_) {
        outputNamesCStr_.push_back(name.c_str());
    }
}

InferenceEngine::Batch InferenceEngine::tokenizeBatch(const std::vector<std::string>& texts, int64_t maxLen) {
    Batch b;
    b.B = static_cast<int64_t>(texts.size());
    b.S = maxLen;
    
    b.input_ids.assign(b.B * b.S, 0);
    b.attention_mask.assign(b.B * b.S, 0);
    b.token_type_ids.assign(b.B * b.S, 0);
    
    for (int64_t i = 0; i < b.B; ++i) {
        std::vector<int32_t> ids = tokenizer_->Encode(texts[i]);
        if (static_cast<int64_t>(ids.size()) > maxLen) {
            ids.resize(maxLen);
        }
        
        for (int64_t t = 0; t < static_cast<int64_t>(ids.size()); ++t) {
            b.input_ids[i * maxLen + t] = static_cast<int64_t>(ids[t]);
            b.attention_mask[i * maxLen + t] = 1;
            b.token_type_ids[i * maxLen + t] = 0;
        }
    }
    
    return b;
}

std::vector<Ort::Value> InferenceEngine::createInputTensors(const Batch& batch) {
    const int64_t B = batch.B;
    const int64_t S = batch.S;
    const std::vector<int64_t> shape{B, S};
    
    std::vector<Ort::Value> ortInputs;
    
    Ort::Value inputIds = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_,
        const_cast<int64_t*>(batch.input_ids.data()),
        batch.input_ids.size(),
        shape.data(),
        shape.size()
    );
    
    Ort::Value attentionMask = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_,
        const_cast<int64_t*>(batch.attention_mask.data()),
        batch.attention_mask.size(),
        shape.data(),
        shape.size()
    );
    
    ortInputs.emplace_back(std::move(inputIds));
    ortInputs.emplace_back(std::move(attentionMask));
    
    if (inputNames_.size() == 3) {
        Ort::Value tokenTypeIds = Ort::Value::CreateTensor<int64_t>(
            memoryInfo_,
            const_cast<int64_t*>(batch.token_type_ids.data()),
            batch.token_type_ids.size(),
            shape.data(),
            shape.size()
        );
        ortInputs.emplace_back(std::move(tokenTypeIds));
    }
    
    return ortInputs;
}

std::vector<float> InferenceEngine::meanPoolL2Norm(const float* lastHidden, const int64_t* mask, 
                                                  int64_t B, int64_t S, int64_t H) {
    std::vector<float> out(B * H, 0.f);
    
    for (int64_t b = 0; b < B; ++b) {
        float count = 0.f;
        for (int64_t t = 0; t < S; ++t) {
            if (mask[b * S + t]) {
                const float* row = lastHidden + (b * S + t) * H;
                for (int64_t h = 0; h < H; ++h) {
                    out[b * H + h] += row[h];
                }
                count += 1.f;
            }
        }
        
        if (count > 0.f) {
            for (int64_t h = 0; h < H; ++h) {
                out[b * H + h] /= count;
            }
        }
        
        double norm = 0.0;
        for (int64_t h = 0; h < H; ++h) {
            norm += out[b * H + h] * out[b * H + h];
        }
        
        norm = std::sqrt(norm) + 1e-12;
        
        for (int64_t h = 0; h < H; ++h) {
            out[b * H + h] = static_cast<float>(out[b * H + h] / norm);
        }
    }
    
    return out;
}

std::vector<uint8_t> InferenceEngine::readFileBytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
}

#ifdef _WIN32
std::wstring InferenceEngine::toWideString(const std::string& s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring ws(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
    while (!ws.empty() && ws.back() == L'\0') {
        ws.pop_back();
    }
    return ws;
}
#endif