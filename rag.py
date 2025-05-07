import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple 

class LocalRAGSystem:
    def __init__(self, model_name='all-mpnet-base-v2'):
        # Initialize sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.index = None
        self.documents = []
        self.metadata = [] 
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.semantic_weight = 0.7 
        self.keyword_weight = 0.3
        
    def chunk_text(self, text, source=None):
        """Split text into overlapping chunks with metadata"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append((chunk, {
                'source': source,
                'start_word': start,
                'end_word': end
            }))
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks
    
    def load_and_process_file(self, file_path):
        """Load text file and process into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks with metadata
        chunks_with_meta = self.chunk_text(text, source=file_path)
        chunks = [chunk for chunk, meta in chunks_with_meta]
        
        self.documents.extend(chunks)
        self.metadata.extend([meta for chunk, meta in chunks_with_meta])
        
        # Generate embeddings for each chunk
        embeddings = self.embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Using Inner Product for similarity
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)
        
        # Update TF-IDF vectors
        self.vectorizer.fit_transform(self.documents)
        
        print(f"Processed {len(chunks)} chunks from {file_path}")
    
    def expand_query(self, query: str) -> List[str]:
        """Generate simple query variations"""
        variations = [query]
        
        # Add basic variations
        if '?' not in query:
            variations.append(query + "?")
        if not query.lower().startswith('what'):
            variations.append("what " + query)
        if not query.lower().startswith('how'):
            variations.append("how " + query)
            
        return variations
    
    def retrieve_docs(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """Retrieve top k relevant documents using hybrid semantic and keyword search"""
        # Generate query variations
        query_variations = self.expand_query(query)
        
        # Semantic search with FAISS for each variation
        semantic_scores = np.zeros(len(self.documents))
        
        for variation in query_variations:
            query_embedding = self.embedding_model.encode([variation])
            query_embedding = np.array(query_embedding).astype('float32')
            query_embedding /= np.linalg.norm(query_embedding)  # Normalize
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(k*2, len(self.documents)))
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:  # FAISS may return -1 for empty indices
                    semantic_scores[idx] = max(semantic_scores[idx], score)
        
        # Keyword search with TF-IDF
        query_vector = self.vectorizer.transform([query])
        doc_vectors = self.vectorizer.transform(self.documents)
        keyword_scores = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Combine scores
        combined_scores = (self.semantic_weight * semantic_scores + 
                          self.keyword_weight * keyword_scores)
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [combined_scores[i] for i in top_indices]
        
        return top_docs, top_scores
    
    def query_ollama(self, context: str, query: str) -> str:
        """Send query to local Ollama instance with improved prompt"""
        try:
            prompt = f"""You are a precise HR assistant that answers questions using ONLY the provided context.
            
            Rules:
            1. If the context contains relevant information, provide a concise answer based solely on that.
            2. If the question asks about something NOT in the context, respond ONLY with: 'I don't have this information in my knowledge base. Please contact hr@greenways.com for assistance.'
            3. Never infer or make up information not explicitly stated in the context.
            4. If the question is ambiguous or unclear, ask for clarification.
            
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more precise answers
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data.get('response', 'No response generated')
                
                # Post-process answer to ensure compliance with rules
                if "I don't know" in answer or "I'm not sure" in answer:
                    return "I don't have this information in my knowledge base. Please contact hr@greenways.com for assistance."
                return answer
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"
    
    def rag_query(self, query: str) -> str:
        print(f"\nQuery: {query}")
        
        # Retrieve relevant documents
        relevant_docs, scores = self.retrieve_docs(query, k=5)
        context = "\n\n".join([f"[Relevance: {score:.2f}] {doc}" 
                             for doc, score in zip(relevant_docs, scores)])
        
        print(f"\nRetrieved Context (top {len(relevant_docs)} chunks):")
        for i, (doc, score) in enumerate(zip(relevant_docs, scores)):
            print(f"\n--- Chunk {i+1} (Score: {score:.2f}) ---")
            print(doc[:500] + ("..." if len(doc) > 500 else ""))
        
        # Generate answer using Ollama
        answer = self.query_ollama("\n".join(relevant_docs), query)
        
        print(f"\nAnswer: {answer}")
        return answer


def main():
    # Initialize RAG system
    rag = LocalRAGSystem()
    
    # Load initial file
    initial_file = "docs/GreenWays_HR_Policies.txt"
    if os.path.exists(initial_file):
        rag.load_and_process_file(initial_file)
    else:
        print(f"⚠️ Initial file not found at {initial_file}")
        print("Please make sure the 'docs' directory exists with the policy file.")
    
    print("\nGreenWays HR Assistant initialized. Type 'quit' to exit.")
    print("You can also load additional files with 'load <filepath>'")
    
    while True:
        try:
            user_input = input("\nHR Query > ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            elif user_input.lower().startswith('load '):
                file_path = user_input[5:].strip()
                if os.path.exists(file_path):
                    rag.load_and_process_file(file_path)
                else:
                    print(f"File not found: {file_path}")
                    
            elif user_input:
                # Test with different phrasings of the same question
                if user_input.lower().startswith('test '):
                    base_question = user_input[5:].strip()
                    variations = [
                        base_question,
                        f"What is the policy about {base_question}?",
                        f"Can you explain {base_question}?",
                        f"Tell me about {base_question}",
                        f"How does {base_question} work?",
                        f"Details on {base_question}"
                    ]
                    for variation in variations:
                        print(f"\nTesting variation: '{variation}'")
                        rag.rag_query(variation)
                else:
                    rag.rag_query(user_input)
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit properly.")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()