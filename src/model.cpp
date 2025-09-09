#include "model.h"


/**
 * Tokenizes a batch of text strings into a structured format for model input.
 * 
 * @param texts Vector of input text strings to tokenize
 * @param tok Pointer to the tokenizer instance
 * @param max_len Maximum sequence length for truncation
 * @return Batch struct containing tokenized input_ids, attention_mask, and token_type_ids
 */
Batch tokenize_batch(
    const std::vector<std::string>& texts,
    Tokenizer* tok,
    int64_t max_len
) {
    Batch b; 

    b.B = (int64_t)texts.size(); 
    b.S = max_len;

    b.input_ids.assign(b.B * b.S, 0);
    b.attention_mask.assign(b.B * b.S, 0);
    b.token_type_ids.assign(b.B * b.S, 0);

    for (int64_t i = 0; i < b.B; ++i) {
        // tokenizers.json carries normalization + post-processing ([CLS]/[SEP]).
        // tokenizers-cpp Encode returns IDs; truncate if longer than max_len.
        std::vector<int32_t> ids = tok->Encode(texts[i]);
        if ((int64_t)ids.size() > max_len) ids.resize(max_len);
            for (int64_t t = 0; t < (int64_t)ids.size(); ++t) {
                b.input_ids[i*max_len + t]      = (int64_t)ids[t];
                b.attention_mask[i*max_len + t] = 1;
                b.token_type_ids[i*max_len + t] = 0;
            }
    }

    return b;
}


/**
 * Performs mean pooling followed by L2 normalization on hidden states.
 * 
 * Reference: https://www.perplexity.ai/search/performs-mean-pooling-followed-afd1ku0AT0u7KAm7LtBy8w
 * 
 * @param last_hidden Pointer to the last hidden layer output (B*S*H)
 * @param mask Pointer to attention mask array (B*S)
 * @param B Batch size
 * @param S Sequence length
 * @param H Hidden dimension size
 * @return Vector of L2-normalized mean-pooled embeddings (B*H)
 */
std::vector<float> mean_pool_l2norm(
    const float* last_hidden,
    const int64_t* mask,
    int64_t B, 
    int64_t S, 
    int64_t H
) {
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
            for (int64_t h = 0; h < H; ++h) 
                out[b*H + h] /= count;
        }
        double nrm = 0.0;
        for (int64_t h = 0; h < H; ++h) 
            nrm += out[b*H + h] * out[b*H + h];

        nrm = std::sqrt(nrm) + 1e-12;

        for (int64_t h = 0; h < H; ++h) 
            out[b*H + h] = float(out[b*H + h] / nrm);
    }

    return out;
}


/**
 * Compute cosine similarity matrix between embeddings
 * 
 * https://www.perplexity.ai/search/what-is-a-cosine-similarity-ma-_l7EyXlbS_.QXzcy7eHycQ
 * 
 * @param E Flattened embedding matrix of shape (B, H) stored row-wise
 * @param B Batch size (number of embeddings)
 * @param H Hidden dimension (embedding size)
 * @return B x B similarity matrix where M[i*B + j] is cosine similarity between embeddings i and j
 * 
 * Note: Assumes input embeddings are L2-normalized, so dot product equals cosine similarity
 */
std::vector<float> cosine_sim_matrix(
    const std::vector<float>& E, 
    int64_t B, 
    int64_t H
) {
    std::vector<float> M(B * B, 0.f);
    for (int64_t i = 0; i < B; ++i) {
        const float* ei = &E[i*H];
        
        for (int64_t j = 0; j < B; ++j) {
            const float* ej = &E[j*H];
            float dot = 0.f;

            for (int64_t h = 0; h < H; ++h) 
                dot += ei[h]*ej[h];
            
            M[i*B + j] = dot; // since rows are L2-normalized, dot = cosine
        }
    }
    return M;
}
