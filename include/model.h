#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include "tokenizers_cpp.h"

using tokenizers::Tokenizer;
typedef std::vector<std::string> string_vec;
typedef std::vector<const char*> cstring_vec;

std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
}

#ifdef _WIN32
std::wstring towstr(const std::string& s) {
  int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring ws(len, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
  while (!ws.empty() && ws.back() == L'\0') ws.pop_back();
  return ws;
}
#endif

// ---------- batching & pooling ----------
struct Batch {
  std::vector<int64_t> input_ids;      // [B*S]
  std::vector<int64_t> attention_mask; // [B*S] (1 for real tokens)
  std::vector<int64_t> token_type_ids; // [B*S] (zeros for single sequence)
  int64_t B{0}, S{0};
};

Batch tokenize_batch(
    const std::vector<std::string>& texts,
    Tokenizer* tok,
    int64_t max_len
);

std::vector<float> mean_pool_l2norm(
    const float* last_hidden,
    const int64_t* mask,
    int64_t B, 
    int64_t S, 
    int64_t H
);

std::vector<float> cosine_sim_matrix(
    const std::vector<float>& E, 
    int64_t B, int64_t H
);


// ---------- i/o discovery ----------
std::tuple<cstring_vec, cstring_vec> discover_io(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t n_inputs = session.GetInputCount();
    string_vec in_names;
    for (size_t i = 0; i < n_inputs; ++i) {
        auto n = session.GetInputNameAllocated(i, allocator);
        in_names.emplace_back(n.get());
    }
    cstring_vec input_names; input_names.reserve(n_inputs);
    for (auto& s : in_names) input_names.push_back(s.c_str());

    auto out_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = out_name_alloc.get(); // usually "last_hidden_state"
    cstring_vec output_names{output_name};

    return {input_names, output_names};
}

std::vector<Ort::Value> make_tensors(
    const Batch& batch,
    std::vector<Ort::Value>& ort_inputs
) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t B = batch.B, S = batch.S;
    const std::vector<int64_t> ishp{B, S};

    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(mem,
        batch.input_ids.data(), batch.input_ids.size(), ishp.data(), ishp.size());
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(mem,
        batch.attention_mask.data(), batch.attention_mask.size(), ishp.data(), ishp.size());

    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ids));
    ort_inputs.emplace_back(std::move(attention_mask));

    if (!batch.token_type_ids.empty()) {
        Ort::Value token_type_ids = Ort::Value::CreateTensor<int64_t>(mem,
            batch.token_type_ids.data(), batch.token_type_ids.size(), ishp.data(), ishp.size());
        ort_inputs.emplace_back(std::move(token_type_ids));
    }

    return ort_inputs;
}

inline std::unique_ptr<Ort::Session> load_model(
    Ort::Env& env,
    const std::string& model_path,
    bool use_cuda
) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        // only if ORT built with CUDA
        OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0);
    }

#ifdef _WIN32
    return std::make_unique<Ort::Session>(env, towstr(model_path).c_str(), so);
#else
    return std::make_unique<Ort::Session>(env, model_path.c_str(), so);
#endif
}

inline std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(const std::string& tok_path) {
    auto blob = read_file_bytes(tok_path);

    std::string json(reinterpret_cast<const char*>(blob.data()), blob.size());
    return tokenizers::Tokenizer::FromBlobJSON(json);
}
