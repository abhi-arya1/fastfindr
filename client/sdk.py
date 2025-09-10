import requests
from typing import Dict, List, Optional, Any

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
    
    def create(self, text: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a new document.
        
        Args:
            text: The document text.
            metadata: Optional metadata dictionary.
            
        Returns:
            Response containing the document ID.
        """
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
        
        response = self.client._request("POST", "/documents", json=payload)
        return response.json()
    
    def retrieve(self, document_id: int) -> Dict[str, Any]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: The document ID.
            
        Returns:
            The document data.
        """
        response = self.client._request("GET", f"/documents/{document_id}")
        return response.json()
    
    def update(self, document_id: int, text: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Update an existing document.
        
        Args:
            document_id: The document ID to update.
            text: The new document text.
            metadata: Optional new metadata.
            
        Returns:
            Response containing update status.
        """
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
        
        response = self.client._request("PUT", f"/documents/{document_id}", json=payload)
        return response.json()
    
    def delete(self, document_id: int) -> Dict[str, Any]:
        """
        Delete a document.
        
        Args:
            document_id: The document ID to delete.
            
        Returns:
            Response containing deletion status.
        """
        response = self.client._request("DELETE", f"/documents/{document_id}")
        return response.json()
    
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
    
    def create_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple documents in a batch.
        
        Args:
            documents: List of document dictionaries with 'text' and optional 'metadata'.
            
        Returns:
            Response containing batch creation status.
        """
        payload = {"documents": documents}
        response = self.client._request("POST", "/documents/batch", json=payload)
        return response.json()


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
              efSearch: int = 350
    ) -> List[Dict[str, Any]]:
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
        
        if efSearch != 350:
            payload["efSearch"] = efSearch
        
        response = self.client._request("POST", "/search", json=payload)
        return response.json()
    
    def semantic(self, query: str, k: int = 10, threshold: float = 0.0, efSearch: int = 350) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: The search query text.
            k: Number of results to return.
            threshold: Minimum score threshold for results (default: 0.0).
            efSearch: HNSW search parameter for performance tuning (default: 350).
            
        Returns:
            List of search results with similarity scores.
        """
        return self.query(query, k=k, type="semantic", threshold=threshold, efSearch=efSearch)
    
    def fulltext(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
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
    
    def by_metadata(self, key: str, value: str, k: int = 10) -> List[Dict[str, Any]]:
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