#!/usr/bin/env python3
"""
File Search App using HelixDB and Google EmbeddingGemma model.
"""

import os
import helix
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Any
from tqdm import tqdm


class FileSearchEngine:
    def __init__(self, db_path: str = "./db", port: int = 6969):
        """Initialize the file search engine."""
        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.db = helix.Client(local=True, port=port, verbose=True)
        self.db_path = db_path
        
    def index_files(self, directory: str, file_extensions: List[str] = None):
        """Index files in the specified directory."""
        if file_extensions is None:
            file_extensions = ['.txt', '.py', '.md', '.js', '.ts', '.json', '.yaml', '.yml', '.rst', '.doc']
            
        indexed_count = 0
        
        for root, dirs, files in tqdm(os.walk(directory)):
            # Skip hidden directories and common build/cache directories
            tqdm.write(f"Indexing directory: {root}")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist']]
            
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Skip empty files
                        if not content.strip():
                            continue
                            
                        # Create embedding using EmbeddingGemma
                        vector = self.model.encode(content).astype(float).tolist()
                        
                        # Insert into HelixDB
                        result = self.db.query("insert_file", {
                            "name": file,
                            "path": file_path,
                            "content": content,
                            "size": os.path.getsize(file_path),
                            "vector": vector
                        })
                        
                        indexed_count += 1
                        print(f"Indexed: {file_path}")
                        
                    except Exception as e:
                        print(f"Error indexing {file_path}: {str(e)}")
                        continue
        
        print(f"\nIndexing complete. Total files indexed: {indexed_count}")
        
    def search_files(self, query: str, k: int = 5) -> List[Dict[Any, Any]]:
        """Search files using semantic similarity."""
        try:
            # Create query embedding
            query_vector = self.model.encode(query).astype(float).tolist()
            
            # Search in HelixDB
            results = self.db.query("search_files", {
                "query_vector": query_vector, 
                "k": k
            })
            
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def search_by_filename(self, filename: str) -> List[Dict[Any, Any]]:
        """Search files by filename using BM25."""
        try:
            results = self.db.query("search_by_name", {
                "filename": filename
            })
            return results
            
        except Exception as e:
            print(f"Filename search error: {str(e)}")
            return []
            
    def display_results(self, results: List[Dict[Any, Any]], query: str):
        """Display search results in a readable format."""
        if not results:
            print(f"No results found for: '{query}'")
            return
            
        print(f"\nSearch results for: '{query}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('name', 'Unknown')}")
            print(f"   Path: {result.get('path', 'Unknown')}")
            print(f"   Size: {result.get('size', 0)} bytes")
            
            # Show preview of content
            content = result.get('content', '')
            if content:
                preview = content[:200].replace('\n', ' ').strip()
                if len(content) > 200:
                    preview += "..."
                print(f"   Preview: {preview}")


def main():
    parser = argparse.ArgumentParser(description="File Search Engine using HelixDB")
    parser.add_argument('command', choices=['index', 'search', 'search-name'], 
                       help='Command to execute')
    parser.add_argument('--directory', '-d', default='.', 
                       help='Directory to index (for index command)')
    parser.add_argument('--query', '-q', 
                       help='Search query (for search commands)')
    parser.add_argument('--limit', '-l', type=int, default=5, 
                       help='Number of results to return')
    parser.add_argument('--extensions', '-e', nargs='+', 
                       help='File extensions to index (e.g., .py .js .md)')
    parser.add_argument('--port', '-p', type=int, default=6969,
                       help='HelixDB port (default: 6969)')
    
    args = parser.parse_args()
    
    # Initialize the search engine
    search_engine = FileSearchEngine(port=args.port)
    
    if args.command == 'index':
        print(f"Starting indexing of directory: {args.directory}")
        if args.extensions:
            print(f"File extensions: {args.extensions}")
        search_engine.index_files(args.directory, args.extensions)
        
    elif args.command == 'search':
        if not args.query:
            print("Error: --query is required for search command")
            sys.exit(1)
        
        results = search_engine.search_files(args.query, args.limit)
        search_engine.display_results(results, args.query)
        
    elif args.command == 'search-name':
        if not args.query:
            print("Error: --query is required for search-name command")
            sys.exit(1)
            
        results = search_engine.search_by_filename(args.query)
        search_engine.display_results(results, args.query)


if __name__ == "__main__":
    main()