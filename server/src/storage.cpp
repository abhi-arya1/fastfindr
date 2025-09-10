#include "storage.h"
#include <iostream>
#include <sstream>
#include <algorithm>

Storage::Storage(const std::string& dbPath) 
    : db_(nullptr), dbPath_(dbPath), inTransaction_(false) {
}

Storage::~Storage() {
    close();
}

bool Storage::initialize() {
    if (db_) {
        return true; // Already initialized
    }
    
    int rc = sqlite3_open(dbPath_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db_) << std::endl;
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    
    // Enable foreign key constraints
    executeSQL("PRAGMA foreign_keys = ON;");
    
    return createTables();
}

void Storage::close() {
    if (db_) {
        if (inTransaction_) {
            rollbackTransaction();
        }
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool Storage::createTables() {
    const std::string createDocumentsTable = R"(
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    )";
    
    const std::string createMetadataTable = R"(
        CREATE TABLE IF NOT EXISTS document_metadata (
            document_id INTEGER,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (document_id, key),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
    )";
    
    const std::string createTextIndex = R"(
        CREATE INDEX IF NOT EXISTS idx_documents_text ON documents(text);
    )";
    
    const std::string createMetadataIndex = R"(
        CREATE INDEX IF NOT EXISTS idx_metadata_key_value ON document_metadata(key, value);
    )";
    
    return executeSQL(createDocumentsTable) &&
           executeSQL(createMetadataTable) &&
           executeSQL(createTextIndex) &&
           executeSQL(createMetadataIndex);
}

size_t Storage::addDocument(const std::string& text, const std::map<std::string, std::string>& metadata) {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return 0;
    }
    
    const std::string sql = "INSERT INTO documents (text) VALUES (?);";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return 0;
    
    sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(stmt);
    finalizeStatement(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Error inserting document: " << sqlite3_errmsg(db_) << std::endl;
        return 0;
    }
    
    size_t documentId = static_cast<size_t>(sqlite3_last_insert_rowid(db_));
    
    // Add metadata
    for (const auto& [key, value] : metadata) {
        if (!addMetadata(documentId, key, value)) {
            std::cerr << "Warning: Failed to add metadata " << key << " for document " << documentId << std::endl;
        }
    }
    
    return documentId;
}

bool Storage::updateDocument(size_t id, const std::string& text, const std::map<std::string, std::string>& metadata) {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return false;
    }
    
    const std::string sql = "UPDATE documents SET text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return false;
    
    sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(id));
    
    int rc = sqlite3_step(stmt);
    finalizeStatement(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Error updating document: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    // Update metadata (delete existing and add new)
    const std::string deleteSql = "DELETE FROM document_metadata WHERE document_id = ?;";
    sqlite3_stmt* deleteStmt = prepareStatement(deleteSql);
    if (deleteStmt) {
        sqlite3_bind_int64(deleteStmt, 1, static_cast<sqlite3_int64>(id));
        sqlite3_step(deleteStmt);
        finalizeStatement(deleteStmt);
    }
    
    for (const auto& [key, value] : metadata) {
        if (!addMetadata(id, key, value)) {
            std::cerr << "Warning: Failed to update metadata " << key << " for document " << id << std::endl;
        }
    }
    
    return sqlite3_changes(db_) > 0;
}

bool Storage::deleteDocument(size_t id) {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return false;
    }
    
    const std::string sql = "DELETE FROM documents WHERE id = ?;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return false;
    
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
    
    int rc = sqlite3_step(stmt);
    finalizeStatement(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Error deleting document: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    return sqlite3_changes(db_) > 0;
}

Document Storage::getDocument(size_t id) {
    Document doc;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return doc;
    }
    
    const std::string sql = "SELECT id, text FROM documents WHERE id = ?;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return doc;
    
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        doc = buildDocumentFromRow(stmt);
        loadDocumentMetadata(doc);
    }
    
    finalizeStatement(stmt);
    return doc;
}

std::vector<Document> Storage::getAllDocuments() {
    std::vector<Document> documents;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return documents;
    }
    
    const std::string sql = "SELECT id, text FROM documents ORDER BY id;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return documents;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Document doc = buildDocumentFromRow(stmt);
        loadDocumentMetadata(doc);
        documents.push_back(doc);
    }
    
    finalizeStatement(stmt);
    return documents;
}

std::vector<Document> Storage::searchDocuments(const std::string& textQuery) {
    std::vector<Document> documents;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return documents;
    }
    
    const std::string sql = "SELECT id, text FROM documents WHERE text LIKE ? ORDER BY id;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return documents;
    
    std::string likePattern = "%" + textQuery + "%";
    sqlite3_bind_text(stmt, 1, likePattern.c_str(), -1, SQLITE_STATIC);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Document doc = buildDocumentFromRow(stmt);
        loadDocumentMetadata(doc);
        documents.push_back(doc);
    }
    
    finalizeStatement(stmt);
    return documents;
}

std::vector<Document> Storage::getDocumentsByMetadata(const std::string& key, const std::string& value) {
    std::vector<Document> documents;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return documents;
    }
    
    const std::string sql = R"(
        SELECT DISTINCT d.id, d.text 
        FROM documents d 
        JOIN document_metadata dm ON d.id = dm.document_id 
        WHERE dm.key = ? AND dm.value = ? 
        ORDER BY d.id;
    )";
    
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return documents;
    
    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_STATIC);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Document doc = buildDocumentFromRow(stmt);
        loadDocumentMetadata(doc);
        documents.push_back(doc);
    }
    
    finalizeStatement(stmt);
    return documents;
}

bool Storage::addMetadata(size_t documentId, const std::string& key, const std::string& value) {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return false;
    }
    
    const std::string sql = "INSERT OR REPLACE INTO document_metadata (document_id, key, value) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return false;
    
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(documentId));
    sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, value.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(stmt);
    finalizeStatement(stmt);
    
    return rc == SQLITE_DONE;
}

bool Storage::updateMetadata(size_t documentId, const std::string& key, const std::string& value) {
    return addMetadata(documentId, key, value); // INSERT OR REPLACE handles updates
}

bool Storage::deleteMetadata(size_t documentId, const std::string& key) {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return false;
    }
    
    const std::string sql = "DELETE FROM document_metadata WHERE document_id = ? AND key = ?;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return false;
    
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(documentId));
    sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_STATIC);
    
    int rc = sqlite3_step(stmt);
    finalizeStatement(stmt);
    
    return rc == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

std::map<std::string, std::string> Storage::getMetadata(size_t documentId) {
    std::map<std::string, std::string> metadata;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return metadata;
    }
    
    const std::string sql = "SELECT key, value FROM document_metadata WHERE document_id = ?;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return metadata;
    
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(documentId));
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string key = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        metadata[key] = value;
    }
    
    finalizeStatement(stmt);
    return metadata;
}

size_t Storage::getDocumentCount() {
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return 0;
    }
    
    const std::string sql = "SELECT COUNT(*) FROM documents;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return 0;
    
    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
    }
    
    finalizeStatement(stmt);
    return count;
}

std::vector<size_t> Storage::getAllDocumentIds() {
    std::vector<size_t> ids;
    if (!db_) {
        std::cerr << "Database not initialized" << std::endl;
        return ids;
    }
    
    const std::string sql = "SELECT id FROM documents ORDER BY id;";
    sqlite3_stmt* stmt = prepareStatement(sql);
    if (!stmt) return ids;
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ids.push_back(static_cast<size_t>(sqlite3_column_int64(stmt, 0)));
    }
    
    finalizeStatement(stmt);
    return ids;
}

bool Storage::beginTransaction() {
    if (!db_ || inTransaction_) {
        return false;
    }
    
    bool success = executeSQL("BEGIN TRANSACTION;");
    if (success) {
        inTransaction_ = true;
    }
    return success;
}

bool Storage::commitTransaction() {
    if (!db_ || !inTransaction_) {
        return false;
    }
    
    bool success = executeSQL("COMMIT;");
    if (success) {
        inTransaction_ = false;
    }
    return success;
}

bool Storage::rollbackTransaction() {
    if (!db_ || !inTransaction_) {
        return false;
    }
    
    bool success = executeSQL("ROLLBACK;");
    if (success) {
        inTransaction_ = false;
    }
    return success;
}

bool Storage::executeSQL(const std::string& sql) {
    if (!db_) {
        return false;
    }
    
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }
    
    return true;
}

sqlite3_stmt* Storage::prepareStatement(const std::string& sql) {
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db_) << std::endl;
        return nullptr;
    }
    
    return stmt;
}

void Storage::finalizeStatement(sqlite3_stmt* stmt) {
    if (stmt) {
        sqlite3_finalize(stmt);
    }
}

Document Storage::buildDocumentFromRow(sqlite3_stmt* stmt) {
    Document doc;
    doc.id = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
    
    const char* textPtr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    if (textPtr) {
        doc.text = textPtr;
    }
    
    return doc;
}

void Storage::loadDocumentMetadata(Document& doc) {
    doc.metadata = getMetadata(doc.id);
}

int Storage::countCallback(void* data, int argc, char** argv, char** azColName) {
    size_t* count = static_cast<size_t*>(data);
    if (argc > 0 && argv[0]) {
        *count = static_cast<size_t>(std::stoull(argv[0]));
    }
    return 0;
}

int Storage::documentCallback(void* data, int argc, char** argv, char** azColName) {
    std::vector<Document>* documents = static_cast<std::vector<Document>*>(data);
    
    if (argc >= 2 && argv[0] && argv[1]) {
        Document doc;
        doc.id = static_cast<size_t>(std::stoull(argv[0]));
        doc.text = argv[1];
        documents->push_back(doc);
    }
    
    return 0;
}