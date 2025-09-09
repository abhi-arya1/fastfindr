#pragma once

#include <string>
#include <vector>
#include <memory>
#include "tokenizers_cpp.h"

class HFTokenizer {
public:
    HFTokenizer();
    ~HFTokenizer();
    
    bool loadTokenizer(const std::string& tokenizer_path);
    std::vector<int64_t> encode(const std::string& text);
    std::string decode(const std::vector<int64_t>& tokens);
    bool isLoaded() const { return tokenizer_ != nullptr; }
    
private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};