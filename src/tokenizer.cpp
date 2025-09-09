#include "tokenizer.h"
#include <iostream>
#include <sstream>

HFTokenizer::HFTokenizer() : tokenizer_(nullptr) {
}

HFTokenizer::~HFTokenizer() {
    tokenizer_.reset();
}

bool HFTokenizer::loadTokenizer(const std::string& tokenizer_path) {
    try {
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_path);
        return tokenizer_ != nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int64_t> HFTokenizer::encode(const std::string& text) {
    if (!tokenizer_) {
        // Fallback to simple tokenization if no tokenizer loaded
        std::vector<int64_t> simple_tokens;
        simple_tokens.push_back(101); // CLS
        
        // Simple word-based tokenization
        std::istringstream iss(text);
        std::string word;
        int64_t token_id = 1000;
        while (iss >> word) {
            simple_tokens.push_back(token_id++);
        }
        
        simple_tokens.push_back(102); // SEP
        return simple_tokens;
    }
    
    auto ids = tokenizer_->Encode(text);
    std::vector<int64_t> token_ids;
    token_ids.reserve(ids.size());
    for (auto id : ids) {
        token_ids.push_back(static_cast<int64_t>(id));
    }
    return token_ids;
}

std::string HFTokenizer::decode(const std::vector<int64_t>& tokens) {
    if (!tokenizer_) {
        return ""; // Can't decode without tokenizer
    }
    
    std::vector<int32_t> int32_tokens;
    int32_tokens.reserve(tokens.size());
    for (auto token : tokens) {
        int32_tokens.push_back(static_cast<int32_t>(token));
    }
    
    return tokenizer_->Decode(int32_tokens);
}