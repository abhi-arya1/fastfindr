#include <iostream>
#include <string>
#include <memory>
#include <filesystem>
#include <signal.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "server.h"
#include "vector_search.h"
#include "util.h"

using json = nlohmann::json;

SearchServer::SearchServer(const ServerConfig& config) : config_(config) {}

bool SearchServer::initialize() {
        std::cout << "Initializing SearchServer..." << std::endl;
        
        // Create new database if requested
        if (config_.create_new_db && std::filesystem::exists(config_.database_path)) {
            std::cout << "Removing existing database..." << std::endl;
            std::filesystem::remove(config_.database_path);
        }
        
        if (config_.create_new_db && std::filesystem::exists(config_.index_path)) {
            std::cout << "Removing existing index..." << std::endl;
            std::filesystem::remove(config_.index_path);
        }

        // Initialize VectorSearch
        vectorSearch_ = std::make_unique<VectorSearch>(
            config_.model_path, 
            config_.tokenizer_path, 
            config_.database_path,
            16, 200
        );

        if (!vectorSearch_->initialize()) {
            std::cerr << "Failed to initialize VectorSearch" << std::endl;
            return false;
        }

        // Load or create index
        vectorSearch_->loadOrCreateIndex(config_.index_path);

        std::cout << "Server initialized with " << vectorSearch_->getDocumentCount() 
                  << " documents" << std::endl;

        setupRoutes();
        return true;
    }

void SearchServer::setupRoutes() {
        // CORS headers
        server_.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // Handle OPTIONS requests
        server_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            return;
        });

        // Health check
        server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "healthy"},
                {"documents", vectorSearch_->getDocumentCount()},
                {"index_size", vectorSearch_->getIndexSize()}
            };
            res.set_content(response.dump(), "application/json");
        });

        // Search endpoint
        server_.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
            handleSearch(req, res);
        });

        // Insert endpoint
        server_.Post("/documents", [this](const httplib::Request& req, httplib::Response& res) {
            handleInsert(req, res);
        });

        // Upsert endpoint (accepts any non-slash characters as ID)
        server_.Put("/documents/([^/]+)", [this](const httplib::Request& req, httplib::Response& res) {
            handleUpsert(req, res);
        });

        // Get by ID endpoint (accepts any non-slash characters as ID)
        server_.Get("/documents/([^/]+)", [this](const httplib::Request& req, httplib::Response& res) {
            handleGetById(req, res);
        });

        // Get by metadata endpoint
        server_.Get("/documents", [this](const httplib::Request& req, httplib::Response& res) {
            handleGetByMetadata(req, res);
        });

        // Delete endpoint (accepts any non-slash characters as ID)
        server_.Delete("/documents/([^/]+)", [this](const httplib::Request& req, httplib::Response& res) {
            handleDelete(req, res);
        });

        // Batch insert endpoint
        server_.Post("/documents/batch", [this](const httplib::Request& req, httplib::Response& res) {
            handleBatchInsert(req, res);
        });

        // Count endpoint
        server_.Get("/documents/count", [this](const httplib::Request& req, httplib::Response& res) {
            handleCount(req, res);
        });

        // Index management endpoints
        server_.Post("/index/rebuild", [this](const httplib::Request& req, httplib::Response& res) {
            vectorSearch_->rebuildIndex();
            vectorSearch_->saveIndex(config_.index_path);
            json response = {{"status", "success"}, {"message", "Index rebuilt"}};
            res.set_content(response.dump(), "application/json");
        });

        server_.Post("/index/save", [this](const httplib::Request& req, httplib::Response& res) {
            vectorSearch_->saveIndex(config_.index_path);
            json response = {{"status", "success"}, {"message", "Index saved"}};
            res.set_content(response.dump(), "application/json");
        });
    }

void SearchServer::handleSearch(const httplib::Request& req, httplib::Response& res) {
        try {
            json request = json::parse(req.body);
            
            if (!request.contains("query")) {
                json error = {{"error", "Missing 'query' field"}};
                res.status = 400;
                res.set_content(error.dump(), "application/json");
                return;
            }

            std::string query = request["query"];
            int k = request.value("k", 10);
            float threshold = request.value("threshold", 0.0f);
            int efSearch = request.value("efSearch", 200);
            
            std::vector<SearchResult> results;
            std::string searchType = request.value("type", "semantic");
            
            if (request.contains("metadata")) {
                // Search by metadata
                auto metadata = request["metadata"];
                if (metadata.contains("key") && metadata.contains("value")) {
                    results = vectorSearch_->searchByMetadata(
                        metadata["key"], metadata["value"], k);
                }
            } else if (searchType == "text" || searchType == "fulltext") {
                // Full text search using SQL
                auto documents = vectorSearch_->getStorage()->searchDocuments(query);
                
                // Convert to SearchResult format and limit results
                for (size_t i = 0; i < documents.size() && i < static_cast<size_t>(k); ++i) {
                    const auto& doc = documents[i];
                    SearchResult result;
                    result.id = doc.id;
                    result.text = doc.text;
                    result.score = 1.0f; // No relevance scoring for text search
                    result.metadata = doc.metadata;
                    // Apply threshold for text search too
                    if (result.score >= threshold) {
                        results.push_back(result);
                    }
                }
            } else {
                // Semantic search (default)
                results = vectorSearch_->searchText(query, k, threshold, efSearch);
            }

            json response = json::array();
            for (const auto& result : results) {
                json doc = {
                    {"id", result.id},
                    {"text", result.text},
                    {"score", result.score},
                    {"metadata", result.metadata}
                };
                response.push_back(doc);
            }

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleInsert(const httplib::Request& req, httplib::Response& res) {
        try {
            json request = json::parse(req.body);
            
            if (!request.contains("text")) {
                json error = {{"error", "Missing 'text' field"}};
                res.status = 400;
                res.set_content(error.dump(), "application/json");
                return;
            }

            std::string text = request["text"];
            std::map<std::string, std::string> metadata;
            std::string customId = request.value("id", "");
            
            if (request.contains("metadata")) {
                for (auto& [key, value] : request["metadata"].items()) {
                    metadata[key] = value.is_string() ? value.get<std::string>() : value.dump();
                }
            }

            std::string documentId = vectorSearch_->addDocument(text, metadata, customId);
            if (documentId.empty()) {
                json error = {{"error", "Failed to insert document. If you provided a custom ID, it may already exist."}};
                res.status = 500;
                res.set_content(error.dump(), "application/json");
                return;
            }

            // Save index after insertion
            vectorSearch_->saveIndex(config_.index_path);

            json response = {
                {"id", documentId},
                {"message", "Document inserted successfully"}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleUpsert(const httplib::Request& req, httplib::Response& res) {
        try {
            std::string id = req.matches[1];
            json request = json::parse(req.body);
            
            if (!request.contains("text")) {
                json error = {{"error", "Missing 'text' field"}};
                res.status = 400;
                res.set_content(error.dump(), "application/json");
                return;
            }

            std::string text = request["text"];
            std::map<std::string, std::string> metadata;
            
            if (request.contains("metadata")) {
                for (auto& [key, value] : request["metadata"].items()) {
                    metadata[key] = value.is_string() ? value.get<std::string>() : value.dump();
                }
            }

            // Use upsert which handles both insert and update
            bool success = vectorSearch_->upsertDocument(id, text, metadata);

            if (!success) {
                json error = {{"error", "Failed to upsert document"}};
                res.status = 500;
                res.set_content(error.dump(), "application/json");
                return;
            }

            // Save index after upsert
            vectorSearch_->saveIndex(config_.index_path);

            json response = {
                {"id", id},
                {"message", "Document upserted successfully"}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleGetById(const httplib::Request& req, httplib::Response& res) {
        try {
            std::string id = req.matches[1];
            Document doc = vectorSearch_->getDocument(id);
            
            if (doc.id.empty()) {
                json error = {{"error", "Document not found"}};
                res.status = 404;
                res.set_content(error.dump(), "application/json");
                return;
            }

            json response = {
                {"id", doc.id},
                {"text", doc.text},
                {"metadata", doc.metadata}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleGetByMetadata(const httplib::Request& req, httplib::Response& res) {
        try {
            std::string key = req.get_param_value("key");
            std::string value = req.get_param_value("value");
            
            if (key.empty() || value.empty()) {
                // Return all documents if no metadata filter
                auto documents = vectorSearch_->getAllDocuments();
                json response = json::array();
                
                for (const auto& doc : documents) {
                    json docJson = {
                        {"id", doc.id},
                        {"text", doc.text},
                        {"metadata", doc.metadata}
                    };
                    response.push_back(docJson);
                }
                
                res.set_content(response.dump(), "application/json");
                return;
            }

            auto documents = vectorSearch_->searchByMetadata(key, value, 1000);
            json response = json::array();
            
            for (const auto& result : documents) {
                json doc = {
                    {"id", result.id},
                    {"text", result.text},
                    {"metadata", result.metadata}
                };
                response.push_back(doc);
            }

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleDelete(const httplib::Request& req, httplib::Response& res) {
        try {
            std::string id = req.matches[1];
            
            bool success = vectorSearch_->deleteDocument(id);
            if (!success) {
                json error = {{"error", "Document not found or failed to delete"}};
                res.status = 404;
                res.set_content(error.dump(), "application/json");
                return;
            }

            // Save index after deletion
            vectorSearch_->saveIndex(config_.index_path);

            json response = {{"message", "Document deleted successfully"}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleCount(const httplib::Request& req, httplib::Response& res) {
        try {
            std::string key = req.get_param_value("key");
            std::string value = req.get_param_value("value");
            
            size_t count = 0;
            
            if (!key.empty() && !value.empty()) {
                // Count documents with specific metadata
                auto documents = vectorSearch_->getStorage()->getDocumentsByMetadata(key, value);
                count = documents.size();
            } else {
                // Count all documents
                count = vectorSearch_->getDocumentCount();
            }
            
            json response = {
                {"count", count}
            };
            
            if (!key.empty() && !value.empty()) {
                response["filter"] = {
                    {"key", key},
                    {"value", value}
                };
            }
            
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::handleBatchInsert(const httplib::Request& req, httplib::Response& res) {
        try {
            json request = json::parse(req.body);
            
            if (!request.contains("documents") || !request["documents"].is_array()) {
                json error = {{"error", "Missing 'documents' array"}};
                res.status = 400;
                res.set_content(error.dump(), "application/json");
                return;
            }

            std::vector<std::string> texts;
            std::vector<std::map<std::string, std::string>> metadataList;
            std::vector<std::string> customIds;
            
            for (const auto& doc : request["documents"]) {
                if (!doc.contains("text")) {
                    json error = {{"error", "Each document must have 'text' field"}};
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                texts.push_back(doc["text"]);
                
                // Check for custom ID
                if (doc.contains("id")) {
                    customIds.push_back(doc["id"]);
                } else {
                    customIds.push_back("");  // Empty string means generate random ID
                }
                
                std::map<std::string, std::string> metadata;
                if (doc.contains("metadata")) {
                    for (auto& [key, value] : doc["metadata"].items()) {
                        metadata[key] = value.is_string() ? value.get<std::string>() : value.dump();
                    }
                }
                metadataList.push_back(metadata);
            }

            vectorSearch_->addDocuments(texts, metadataList, customIds);
            
            // Save index after batch insertion
            vectorSearch_->saveIndex(config_.index_path);

            json response = {
                {"message", "Documents inserted successfully"},
                {"count", texts.size()}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    }

void SearchServer::run() {
    std::cout << "Starting server on " << config_.host << ":" << config_.port << std::endl;
    server_.listen(config_.host.c_str(), config_.port);
}

void SearchServer::stop() {
    server_.stop();
}

ServerConfig parseServerOptions(int argc, char** argv) {
    ServerConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            config.tokenizer_path = argv[++i];
        } else if (arg == "--database" && i + 1 < argc) {
            config.database_path = argv[++i];
        } else if (arg == "--index" && i + 1 < argc) {
            config.index_path = argv[++i];
        } else if (arg == "--new-db") {
            config.create_new_db = true;
        } else if (arg == "--level" && i + 1 < argc) {
            int level = std::stoi(argv[++i]);
            switch (level) {
                case 1: config.logging_level = ORT_LOGGING_LEVEL_WARNING; break;
                case 2: config.logging_level = ORT_LOGGING_LEVEL_INFO; break;
                case 3: config.logging_level = ORT_LOGGING_LEVEL_VERBOSE; break;
                default: std::cout << "Invalid log level. Using default (INFO)." << std::endl;
            }
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --host HOST         Server host (default: localhost)\n";
            std::cout << "  --port PORT         Server port (default: 8080)\n";
            std::cout << "  --model PATH        Path to ONNX model file\n";
            std::cout << "  --tokenizer PATH    Path to tokenizer file\n";
            std::cout << "  --database PATH     Path to SQLite database file\n";
            std::cout << "  --index PATH        Path to FAISS index file\n";
            std::cout << "  --new-db            Create new database (removes existing)\n";
            std::cout << "  --level LEVEL       Logging level (1=WARNING, 2=INFO, 3=VERBOSE)\n";
            std::cout << "  --help              Show this help message\n";
            exit(0);
        }
    }
    
    return config;
}

int main(int argc, char** argv) {
    ServerConfig config = parseServerOptions(argc, argv);
    
    std::cout << "Starting Semantic Search Server..." << std::endl;
    std::cout << "Model: " << config.model_path << std::endl;
    std::cout << "Tokenizer: " << config.tokenizer_path << std::endl;
    std::cout << "Database: " << config.database_path << std::endl;
    std::cout << "Index: " << config.index_path << std::endl;
    
    SearchServer server(config);
    
    if (!server.initialize()) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }
    
    // Handle Ctrl+C gracefully
    signal(SIGINT, [](int) {
        std::cout << "\nShutting down server..." << std::endl;
        exit(0);
    });
    
    server.run();
    
    return 0;
}