#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sqlite3.h>

struct Document {
    std::string id;
    std::string text;
    std::map<std::string, std::string> metadata;
    
    Document() {}
    Document(const std::string& doc_id, const std::string& doc_text, 
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
    
    std::string addDocument(const std::string& text, const std::map<std::string, std::string>& metadata = {}, const std::string& customId = "");
    bool updateDocument(const std::string& id, const std::string& text, const std::map<std::string, std::string>& metadata = {});
    bool upsertDocument(const std::string& id, const std::string& text, const std::map<std::string, std::string>& metadata = {});
    bool deleteDocument(const std::string& id);
    
    Document getDocument(const std::string& id);
    std::vector<Document> getAllDocuments();
    std::vector<Document> searchDocuments(const std::string& textQuery);
    std::vector<Document> getDocumentsByMetadata(const std::string& key, const std::string& value);
    
    bool addMetadata(const std::string& documentId, const std::string& key, const std::string& value);
    bool updateMetadata(const std::string& documentId, const std::string& key, const std::string& value);
    bool deleteMetadata(const std::string& documentId, const std::string& key);
    std::map<std::string, std::string> getMetadata(const std::string& documentId);
    
    size_t getDocumentCount();
    std::vector<std::string> getAllDocumentIds();
    bool documentExists(const std::string& id);
    
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
    std::string generateRandomId();
    
    Document buildDocumentFromRow(sqlite3_stmt* stmt);
    void loadDocumentMetadata(Document& doc);
    
    static int countCallback(void* data, int argc, char** argv, char** azColName);
    static int documentCallback(void* data, int argc, char** argv, char** azColName);
};