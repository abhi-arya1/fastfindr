// C++17
// deps: ONNX Runtime C++ API + tokenizers-cpp (tiny wrapper around HF tokenizers)
//      https://github.com/mlc-ai/tokenizers-cpp
// build: see CMakeLists.txt below

#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <stdexcept>

// tokenizers-cpp (header in your include path)
#include "tokenizers_cpp.h"

using tokenizers::Tokenizer;

// ---------- I/O helpers ----------
static std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("Cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
}

#ifdef _WIN32
static std::wstring towstr(const std::string& s) {
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

static Batch tokenize_batch(const std::vector<std::string>& texts,
                            Tokenizer* tok,
                            int64_t max_len) {
  Batch b; b.B = (int64_t)texts.size(); b.S = max_len;
  b.input_ids.assign(b.B * b.S, 0);
  b.attention_mask.assign(b.B * b.S, 0);
  b.token_type_ids.assign(b.B * b.S, 0);

  for (int64_t i = 0; i < b.B; ++i) {
    // tokenizers.json carries normalization + post-processing ([CLS]/[SEP]).
    // tokenizers-cpp Encode returns IDs; truncate if longer than max_len.
    std::vector<int> ids = tok->Encode(texts[i]);
    if ((int64_t)ids.size() > max_len) ids.resize(max_len);
    for (int64_t t = 0; t < (int64_t)ids.size(); ++t) {
      b.input_ids[i*max_len + t]      = (int64_t)ids[t];
      b.attention_mask[i*max_len + t] = 1;
      b.token_type_ids[i*max_len + t] = 0;
    }
  }
  return b;
}

// mask-aware mean pooling + L2 norm
static std::vector<float> mean_pool_l2norm(const float* last_hidden,
                                           const int64_t* mask,
                                           int64_t B, int64_t S, int64_t H) {
  std::vector<float> out(B * H, 0.f);
  for (int64_t b = 0; b < B; ++b) {
    float count = 0.f;
    for (int64_t t = 0; t < S; ++t) {
      if (mask[b*S + t]) {
        const float* row = last_hidden + (b*S + t) * H;
        for (int64_t h = 0; h < H; ++h) out[b*H + h] += row[h];
        count += 1.f;
      }
    }
    if (count > 0.f) {
      for (int64_t h = 0; h < H; ++h) out[b*H + h] /= count;
    }
    double nrm = 0.0;
    for (int64_t h = 0; h < H; ++h) nrm += out[b*H + h] * out[b*H + h];
    nrm = std::sqrt(nrm) + 1e-12;
    for (int64_t h = 0; h < H; ++h) out[b*H + h] = float(out[b*H + h] / nrm);
  }
  return out;
}

// cosine similarity matrix for L2-normalized embeddings
static std::vector<float> cosine_sim_matrix(const std::vector<float>& E, int64_t B, int64_t H) {
  std::vector<float> M(B * B, 0.f);
  for (int64_t i = 0; i < B; ++i) {
    const float* ei = &E[i*H];
    for (int64_t j = 0; j < B; ++j) {
      const float* ej = &E[j*H];
      float dot = 0.f;
      for (int64_t h = 0; h < H; ++h) dot += ei[h]*ej[h];
      M[i*B + j] = dot; // since rows are L2-normalized, dot = cosine
    }
  }
  return M;
}

int main(int argc, char** argv) {
  const std::string model_path = (argc > 1) ? argv[1] : "embeddinggemma-onnx/model.onnx";
  const std::string tok_path   = (argc > 2) ? argv[2] : "embeddinggemma-onnx/tokenizer.json";
  const int64_t MAX_LEN = 256; // keep ≤ model's max_position_embeddings

  // 1) Load tokenizer
  auto tok_blob = read_file_bytes(tok_path);
  auto tok = Tokenizer::FromBlobJSON(tok_blob);

  // 2) Test sentences (same as your Python example)
  std::vector<std::string> sentences = {
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
  };
  Batch batch = tokenize_batch(sentences, tok.get(), MAX_LEN);

  // 3) Create ONNX Runtime session
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "st");
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // For CUDA, if you built ORT with CUDA: OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0);

#ifdef _WIN32
  Ort::Session session(env, towstr(model_path).c_str(), so);
#else
  Ort::Session session(env, model_path.c_str(), so);
#endif

  // 4) Discover inputs/outputs
  Ort::AllocatorWithDefaultOptions allocator;
  size_t n_inputs = session.GetInputCount();
  std::vector<std::string> in_names;
  for (size_t i = 0; i < n_inputs; ++i) {
    auto n = session.GetInputNameAllocated(i, allocator);
    in_names.emplace_back(n.get());
  }
  std::vector<const char*> input_names; input_names.reserve(n_inputs);
  for (auto& s : in_names) input_names.push_back(s.c_str());

  auto out_name_alloc = session.GetOutputNameAllocated(0, allocator);
  const char* output_name = out_name_alloc.get(); // usually "last_hidden_state"
  std::vector<const char*> output_names{output_name};

  // 5) Make input tensors
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t B = batch.B, S = batch.S;
  const std::vector<int64_t> ishp{B, S};

  Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(mem,
    batch.input_ids.data(), batch.input_ids.size(), ishp.data(), ishp.size());
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(mem,
    batch.attention_mask.data(), batch.attention_mask.size(), ishp.data(), ishp.size());

  std::vector<Ort::Value> ort_inputs;
  if (n_inputs == 3) {
    Ort::Value token_type_ids = Ort::Value::CreateTensor<int64_t>(mem,
      batch.token_type_ids.data(), batch.token_type_ids.size(), ishp.data(), ishp.size());
    ort_inputs.emplace_back(std::move(input_ids));
    ort_inputs.emplace_back(std::move(attention_mask));
    ort_inputs.emplace_back(std::move(token_type_ids));
  } else {
    ort_inputs.emplace_back(std::move(input_ids));
    ort_inputs.emplace_back(std::move(attention_mask));
  }

  // 6) Run model
  auto ort_outputs = session.Run(Ort::RunOptions{nullptr},
                                 input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names.data(), output_names.size());

  auto& last_hidden_val = ort_outputs.front();
  auto shape_info = last_hidden_val.GetTensorTypeAndShapeInfo();
  auto dims = shape_info.GetShape(); // expect [B, S, H]
  if (dims.size() != 3) throw std::runtime_error("Unexpected output rank; expected 3.");
  int64_t H = dims[2];
  const float* last_hidden = last_hidden_val.GetTensorData<float>();

  // 7) Pool -> embeddings
  std::vector<float> embeddings = mean_pool_l2norm(last_hidden, batch.attention_mask.data(), B, S, H);

  // 8) Cosine similarities (B x B)
  std::vector<float> sims = cosine_sim_matrix(embeddings, B, H);

  // 9) Print shape + matrix
  std::cout << "[" << B << ", " << B << "]\n";
  for (int64_t i = 0; i < B; ++i) {
    for (int64_t j = 0; j < B; ++j) {
      std::cout << sims[i*B + j] << (j+1<B ? " " : "");
    }
    std::cout << "\n";
  }
  return 0;
}