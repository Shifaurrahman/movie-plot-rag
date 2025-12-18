"""
Mini RAG System for Movie Plots
A lightweight retrieval-augmented generation system using FAISS and OpenAI
"""

import json
import re
from typing import List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (API key loaded from .env)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    movie_title: str
    chunk_id: int


class MoviePlotRAG:
    """Lightweight RAG system for movie plot queries"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.index = None
        self.dimension = 1536  # text-embedding-3-small dimension
        
    def load_and_preprocess(self, csv_path: str, sample_size: int = 300) -> None:
        """Load dataset and sample rows"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Sample and clean
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        df = df[['Title', 'Plot']].dropna()
        
        print(f"Loaded {len(df)} movies")
        
        # Chunk the plots
        for _, row in df.iterrows():
            chunks = self._chunk_text(row['Plot'], row['Title'])
            self.chunks.extend(chunks)
            
        print(f"Created {len(self.chunks)} chunks")
    
    def _chunk_text(self, text: str, title: str, max_words: int = 300) -> List[Chunk]:
        """Split text into chunks of ~max_words"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            chunks.append(Chunk(
                text=chunk_text,
                movie_title=title,
                chunk_id=i // max_words
            ))
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI"""
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def build_index(self) -> None:
        """Build FAISS index from chunks"""
        print("Building FAISS index...")
        
        # Get embeddings for all chunks
        embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 50 == 0:
                print(f"Embedding chunk {i}/{len(self.chunks)}")
            emb = self._get_embedding(chunk.text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 3) -> List[Chunk]:
        """Retrieve top-k relevant chunks"""
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding])
        
        distances, indices = self.index.search(query_embedding, k)
        
        return [self.chunks[idx] for idx in indices[0]]
    
    def generate_answer(self, query: str, k: int = 3) -> Dict:
        """Generate structured answer using retrieved context"""
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, k)
        
        # Format context
        context = "\n\n".join([
            f"Movie: {chunk.movie_title}\nPlot excerpt: {chunk.text}"
            for chunk in retrieved_chunks
        ])
        
        # Create prompt
        prompt = f"""You are a movie expert. Answer the question based on the provided plot excerpts.

Question: {query}

Retrieved plot excerpts:
{context}

Provide a clear, concise answer based on the information above. If the information doesn't contain a clear answer, say so."""

        # Generate answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful movie knowledge assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Format contexts for output
        contexts = [
            f"{chunk.movie_title}: {chunk.text[:200]}..."
            for chunk in retrieved_chunks
        ]
        
        # Generate reasoning
        reasoning = f"Retrieved {k} relevant plot excerpts. The answer was formed by analyzing mentions of the query topic across these movies: {', '.join([c.movie_title for c in retrieved_chunks])}."
        
        return {
            "answer": answer,
            "contexts": contexts,
            "reasoning": reasoning
        }


def main():
    """Main execution function"""
    import sys
    
    # Check for CSV path
    if len(sys.argv) < 2:
        print("Usage: python movie_rag.py <path_to_csv>")
        print("Example: python movie_rag.py wiki_movie_plots_deduped.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Initialize RAG system
    rag = MoviePlotRAG()
    
    # Load and process data
    rag.load_and_preprocess(csv_path, sample_size=300)
    
    # Build index
    rag.build_index()
    
    # Example queries
    queries = [
        "Which movies feature artificial intelligence?",
        "Tell me about movies with time travel",
        "What movies involve romantic storylines in Paris?"
    ]
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES")
    print("="*80 + "\n")
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        result = rag.generate_answer(query, k=3)
        
        # Pretty print JSON
        print(json.dumps(result, indent=2))
        print("\n")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Enter your own queries (or 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        user_query = input("\nYour question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_query:
            continue
        
        result = rag.generate_answer(user_query, k=3)
        print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()