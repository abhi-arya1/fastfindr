#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sqlite3.h>

struct Document {
    size_t id;
    std::string text;
    std::map<std::string, std::string> metadata;
    
    Document() : id(0) {}
    Document(size_t doc_id, const std::string& doc_text, 
             const std::map<std::string, std::string>& doc_metadata = {})
        : id(doc_id), text(doc_text), metadata(doc_metadata) {}
};

class Storage {
public:
    explicit Storage(const std::string& dbPath);
    ~Storage();
    
    bool initialize();
    void close();
    bool isOpen() const { return db_ != nullptr; }
    
    size_t addDocument(const std::string& text, const std::map<std::string, std::string>& metadata = {});
    bool updateDocument(size_t id, const std::string& text, const std::map<std::string, std::string>& metadata = {});
    bool deleteDocument(size_t id);
    
    Document getDocument(size_t id);
    std::vector<Document> getAllDocuments();
    std::vector<Document> searchDocuments(const std::string& textQuery);
    std::vector<Document> getDocumentsByMetadata(const std::string& key, const std::string& value);
    
    bool addMetadata(size_t documentId, const std::string& key, const std::string& value);
    bool updateMetadata(size_t documentId, const std::string& key, const std::string& value);
    bool deleteMetadata(size_t documentId, const std::string& key);
    std::map<std::string, std::string> getMetadata(size_t documentId);
    
    size_t getDocumentCount();
    std::vector<size_t> getAllDocumentIds();
    
    bool beginTransaction();
    bool commitTransaction();
    bool rollbackTransaction();

private:
    sqlite3* db_;
    std::string dbPath_;
    bool inTransaction_;
    
    bool createTables();
    bool executeSQL(const std::string& sql);
    sqlite3_stmt* prepareStatement(const std::string& sql);
    void finalizeStatement(sqlite3_stmt* stmt);
    
    Document buildDocumentFromRow(sqlite3_stmt* stmt);
    void loadDocumentMetadata(Document& doc);
    
    static int countCallback(void* data, int argc, char** argv, char** azColName);
    static int documentCallback(void* data, int argc, char** argv, char** azColName);
};