#!/usr/bin/env python3
"""
Create a sample RAGpack for testing the retriever.
"""

import json
import zipfile
import numpy as np
from pathlib import Path
from io import BytesIO

def create_sample_ragpack():
    """Create a sample RAGpack for demonstration."""
    
    # Sample chunks about AI/ML topics
    chunks = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to automatically learn representations from data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions by receiving rewards or penalties for their actions.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds patterns in data without explicit target labels or supervision.",
        "Transfer learning leverages pre-trained models to solve related tasks with less data and computation.",
        "Neural networks are inspired by biological neurons and consist of interconnected processing units.",
        "Artificial intelligence aims to create systems that can perform tasks that typically require human intelligence."
    ]
    
    # Create simple embeddings (random but consistent)
    np.random.seed(42)
    embeddings = np.random.randn(len(chunks), 384).astype('float32')
    
    # Create metadata
    metadata = {
        "original_pdf": "ai_ml_concepts.pdf",
        "chunk_size": 150,
        "model_used": "all-MiniLM-L6-v2",
        "timestamp": "2025-01-01T12:00:00"
    }
    
    # Create zip file
    zip_path = Path("sample_ragpack.zip")
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Add chunks
        chunks_json = json.dumps(chunks, ensure_ascii=False).encode('utf-8')
        zf.writestr("chunks.json", chunks_json)
        
        # Add embeddings (npy format)
        embeddings_bytes_io = BytesIO()
        np.save(embeddings_bytes_io, embeddings)
        zf.writestr("embeddings.npy", embeddings_bytes_io.getvalue())
        
        # Add embeddings (csv format)
        embeddings_csv_io = BytesIO()
        np.savetxt(embeddings_csv_io, embeddings, delimiter=",")
        zf.writestr("embeddings.csv", embeddings_csv_io.getvalue())
        
        # Add metadata
        metadata_json = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        zf.writestr("metadata.json", metadata_json)
    
    print(f"Sample RAGpack created: {zip_path}")
    print(f"Contains {len(chunks)} chunks about AI/ML topics")
    return zip_path

if __name__ == "__main__":
    create_sample_ragpack()