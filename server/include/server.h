#pragma once

#include <string>
#include <memory>
#include <httplib.h>
#include <onnxruntime_cxx_api.h>

class VectorSearch;

struct ServerConfig {
    std::string host = "localhost";
    int port = 8080;
    std::string model_path = "../embeddinggemma-onnx/model.onnx";
    std::string tokenizer_path = "../embeddinggemma-onnx/tokenizer.json";
    std::string database_path = "database.db";
    std::string index_path = "vectors.index";
    bool create_new_db = false;
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_INFO;
};

class SearchServer {
private:
    std::unique_ptr<VectorSearch> vectorSearch_;
    ServerConfig config_;
    httplib::Server server_;

public:
    SearchServer(const ServerConfig& config);
    
    bool initialize();
    void setupRoutes();
    void run();
    void stop();

private:
    void handleSearch(const httplib::Request& req, httplib::Response& res);
    void handleInsert(const httplib::Request& req, httplib::Response& res);
    void handleUpsert(const httplib::Request& req, httplib::Response& res);
    void handleGetById(const httplib::Request& req, httplib::Response& res);
    void handleGetByMetadata(const httplib::Request& req, httplib::Response& res);
    void handleDelete(const httplib::Request& req, httplib::Response& res);
    void handleBatchInsert(const httplib::Request& req, httplib::Response& res);
    void handleTextSearch(const httplib::Request& req, httplib::Response& res);
    void handleCount(const httplib::Request& req, httplib::Response& res);
};

ServerConfig parseServerOptions(int argc, char** argv);