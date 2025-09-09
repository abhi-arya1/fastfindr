#include <iostream>
#include "util.h"
#include "model.h"
#include "vector_search.h"
#include "embedding_loader.h"


int main(int argc, char** argv) {
    RuntimeOptions options = parse_runtime_options(argc, argv);

    Ort::Env env(options.logging_level, "fastfindr");
    std::cout << "Hello, World!" << std::endl;

    const std::string model_path = "embeddinggemma-onnx/model.onnx";
    const std::string tokenizer_path = "embeddinggemma-onnx/tokenizer.json";

    auto tokenizer = load_tokenizer(tokenizer_path);
    auto model = load_model(env, model_path, options.use_cuda);

    

    return 0;
}
