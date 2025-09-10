import requests
from typing import Dict, List, Optional, Any, TypedDict


class Document(TypedDict):
    id: str
    text: str
    metadata: Optional[Dict[str, str]]

class Client:
    """OpenAI-style client for the semantic search API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the client.
        
        Args:
            base_url: The base URL of the server.
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Create sub-clients
        self.documents = DocumentsClient(self)
        self.search = SearchClient(self)
        self.index = IndexClient(self)
    
    def health(self) -> Dict[str, Any]:
        """Check server health and get statistics."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Internal method to make HTTP requests."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response


class DocumentsClient:
    """Client for document operations."""

    def __init__(self, client: Client):
        self.client = client
    
    @property
    def total_count(self) -> int:
        """Get the total count of documents in the database."""
        return self.count()
    
    def create(self, text: str, metadata: Optional[Dict[str, str]] = None, id: Optional[str] = None) -> Document:
        """
        Create a new document.
        
        Args:
            text: The document text.
            metadata: Optional metadata dictionary.
            id: Optional custom ID for the document (e.g., file path).
            
        Returns:
            Response containing the document ID.
        """
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
        if id:
            payload["id"] = id
        
        response = self.client._request("POST", "/documents", json=payload)
        return response.json()
    
    def retrieve(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: The document ID (string).
            
        Returns:
            The document data.
        """
        # URL encode the ID to handle special characters like paths
        from urllib.parse import quote
        encoded_id = quote(str(document_id), safe='')
        response = self.client._request("GET", f"/documents/{encoded_id}")
        return response.json()
    
    def update(self, document_id: str, text: str, metadata: Optional[Dict[str, str]] = None) -> Document:
        """
        Update an existing document.
        
        Args:
            document_id: The document ID to update (string).
            text: The new document text.
            metadata: Optional new metadata.
            
        Returns:
            Response containing update status.
        """
        from urllib.parse import quote
        encoded_id = quote(str(document_id), safe='')
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
        
        response = self.client._request("PUT", f"/documents/{encoded_id}", json=payload)
        return response.json()
    
    def upsert(self, document_id: str, text: str, metadata: Optional[Dict[str, str]] = None) -> Document:
        """
        Upsert a document (insert if doesn't exist, update if exists).
        
        Args:
            document_id: The document ID (string).
            text: The document text.
            metadata: Optional metadata.
            
        Returns:
            Response containing upsert status.
        """
        from urllib.parse import quote
        encoded_id = quote(str(document_id), safe='')
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
        
        response = self.client._request("PUT", f"/documents/{encoded_id}", json=payload)
        return response.json()
    
    def delete(self, document_id: str) -> Document:
        """
        Delete a document.
        
        Args:
            document_id: The document ID to delete (string).
            
        Returns:
            Response containing deletion status.
        """
        from urllib.parse import quote
        encoded_id = quote(str(document_id), safe='')
        response = self.client._request("DELETE", f"/documents/{encoded_id}")
        return response.json()
    
    def count(self, key: Optional[str] = None, value: Optional[str] = None) -> int:
        """
        Count documents, optionally filtered by metadata.
        
        Args:
            key: Optional metadata key to filter by.
            value: Optional metadata value to filter by.
            
        Returns:
            Count of documents.
        """
        params = {}
        if key and value:
            params = {"key": key, "value": value}
        
        response = self.client._request("GET", "/documents/count", params=params)
        return response.json().get("count", 0)
    
    def list(self, key: Optional[str] = None, value: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List documents, optionally filtered by metadata.
        
        Args:
            key: Optional metadata key to filter by.
            value: Optional metadata value to filter by.
            
        Returns:
            List of documents.
        """
        params = {}
        if key and value:
            params = {"key": key, "value": value}
        
        response = self.client._request("GET", "/documents", params=params)
        return response.json()
    
    def create_batch(self, documents: List[Document]) -> List[Document]:
        """
        Create multiple documents in a batch.
        
        Args:
            documents: List of document dictionaries with 'text', optional 'metadata', and optional 'id'.
                      Example: [{'text': 'content', 'metadata': {'key': 'value'}, 'id': 'custom_id'}]
            
        Returns:
            Response containing batch creation status.
        """
        payload = {"documents": documents}
        response = self.client._request("POST", "/documents/batch", json=payload)
        return response.json()
    
    def upsert_batch(self, documents: List[Document]) -> List[Document]:
        """
        Upsert multiple documents (insert or update each).
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and optional 'metadata'.
                      The 'id' field is required for upsert.
            
        Returns:
            List of responses for each upsert operation.
        """
        results = []
        for doc in documents:
            if 'id' not in doc:
                results.append({"error": "Document missing required 'id' field for upsert"})
                continue
            
            result = self.upsert(
                document_id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata')
            )
            results.append(result)
        
        return results


class SearchClient:
    """Client for search operations."""

    def __init__(self, client: Client):
        self.client = client
    
    def query(self, 
              query: str, 
              k: int = 10, 
              type: str = "semantic",
              metadata: Optional[Dict[str, str]] = None,
              threshold: float = 0.0,
              efSearch: int = 200
    ) -> List[Document]:
        """
        Search for documents.
        
        Args:
            query: The search query text.
            k: Number of results to return (default: 10).
            type: Search type - "semantic" (default), "text", or "fulltext".
            metadata: Optional metadata filter with 'key' and 'value'.
            threshold: Minimum score threshold for results (default: 0.0).
            efSearch: HNSW search parameter for performance tuning (default: 350).
            
        Returns:
            List of search results with scores.
        """
        payload = {
            "query": query,
            "k": k
        }
        
        if type != "semantic":
            payload["type"] = type
        
        if metadata:
            payload["metadata"] = metadata
        
        if threshold != 0.0:
            payload["threshold"] = threshold
        
        if efSearch != 200:
            payload["efSearch"] = efSearch
        
        response = self.client._request("POST", "/search", json=payload)
        return response.json()
    
    def semantic(self, query: str, k: int = 10, threshold: float = 0.0, efSearch: int = 350) -> List[Document]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: The search query text.
            k: Number of results to return.
            threshold: Minimum score threshold for results (default: 0.0).
            efSearch: HNSW search parameter for performance tuning (default: 200).
            
        Returns:
            List of search results with similarity scores.
        """
        return self.query(query, k=k, type="semantic", threshold=threshold, efSearch=efSearch)
    
    def fulltext(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Document]:
        """
        Perform full-text search using SQL LIKE.
        
        Args:
            query: The search query text.
            k: Number of results to return.
            threshold: Minimum score threshold for results (default: 0.0).
            
        Returns:
            List of search results.
        """
        return self.query(query, k=k, type="text", threshold=threshold)
    
    def by_metadata(self, key: str, value: str, k: int = 10) -> List[Document]:
        """
        Search documents by metadata key-value pair.
        
        Args:
            key: The metadata key to search.
            value: The metadata value to match.
            k: Number of results to return.
            
        Returns:
            List of matching documents.
        """
        return self.query("", k=k, metadata={"key": key, "value": value})


class IndexClient:
    """Client for index management operations."""

    def __init__(self, client: Client):
        self.client = client
    
    def rebuild(self) -> Dict[str, Any]:
        """
        Rebuild the vector index.
        
        Returns:
            Response containing rebuild status.
        """
        response = self.client._request("POST", "/index/rebuild")
        return response.json()
    
    def save(self) -> Dict[str, Any]:
        """
        Save the current index to disk.
        
        Returns:
            Response containing save status.
        """
        response = self.client._request("POST", "/index/save")
        return response.json()