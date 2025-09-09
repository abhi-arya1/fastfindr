#include "embedding_loader.h"
#include <iostream>
#include <stdexcept>

EmbeddingLoader::EmbeddingLoader() 
    : memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , embeddingDim_(0)
    , loaded_(false) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EmbeddingLoader");
}

EmbeddingLoader::~EmbeddingLoader() {
    unloadModel();
}

bool EmbeddingLoader::loadModel(const std::string& modelPath) {
    try {
        initializeSession(modelPath);
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

void EmbeddingLoader::unloadModel() {
    if (session_) {
        session_.reset();
        inputNames_.clear();
        outputNames_.clear();
        inputShapes_.clear();
        outputShapes_.clear();
        embeddingDim_ = 0;
        loaded_ = false;
        std::cout << "Model unloaded." << std::endl;
    }
}

bool EmbeddingLoader::isLoaded() const {
    return loaded_;
}

void EmbeddingLoader::initializeSession(const std::string& modelPath) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
}

void EmbeddingLoader::extractModelInfo() {
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input info
    size_t numInputNodes = session_->GetInputCount();
    inputNames_.reserve(numInputNodes);
    inputShapes_.reserve(numInputNodes);
    
    for (size_t i = 0; i < numInputNodes; i++) {
        auto inputName = session_->GetInputNameAllocated(i, allocator);
        inputNames_.push_back(inputName.get());
        
        Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = inputTensorInfo.GetShape();
        inputShapes_.push_back(inputShape);
    }
    
    // Get output info
    size_t numOutputNodes = session_->GetOutputCount();
    outputNames_.reserve(numOutputNodes);
    outputShapes_.reserve(numOutputNodes);
    
    for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.push_back(outputName.get());
        
        Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        outputShapes_.push_back(outputShape);
        
        // Assume the embedding dimension is the last dimension of the first output
        if (i == 0 && !outputShape.empty()) {
            embeddingDim_ = static_cast<size_t>(outputShape.back());
        }
    }
}

std::vector<float> EmbeddingLoader::getEmbedding(const std::vector<int64_t>& inputIds) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(inputIds.size())};
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo_, 
        const_cast<int64_t*>(inputIds.data()), 
        inputIds.size(), 
        inputShape.data(), 
        inputShape.size()
    );
    
    // Run inference
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, 
        inputNames_.data(), 
        &inputTensor, 
        1, 
        outputNames_.data(), 
        outputNames_.size()
    );
    
    // Extract output
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t outputSize = 1;
    for (auto dim : outputShape) {
        outputSize *= static_cast<size_t>(dim);
    }
    
    return std::vector<float>(outputData, outputData + outputSize);
}

std::vector<std::vector<float>> EmbeddingLoader::getEmbeddings(const std::vector<std::vector<int64_t>>& batchInputIds) {
    std::vector<std::vector<float>> results;
    results.reserve(batchInputIds.size());
    
    for (const auto& inputIds : batchInputIds) {
        results.push_back(getEmbedding(inputIds));
    }
    
    return results;
}