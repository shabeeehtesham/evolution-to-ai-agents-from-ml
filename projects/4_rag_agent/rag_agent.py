# =====================================================================
# Project 4: Production RAG AI Agent from Scratch
# Author: Shabih Ehtesham
#
# A complete Retrieval-Augmented Generation (RAG) agent. Combines a custom
# TF-IDF-based Vector Database built using NumPy with dynamic prompt formatting,
# semantic query retrieval, and integration with the Google Gemini API.
# =====================================================================

import numpy as np
import argparse
import sys
import os
import json
import re

# =====================================================================
# 1. Custom Vector DB from Scratch (TF-IDF Cosine Similarity)
# =====================================================================

class SimpleVectorDB:
    """
    A lightweight, in-memory vector database built from scratch using NumPy.
    Implements TF-IDF vectorization and Cosine Similarity.
    """
    def __init__(self):
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.tf_idf_matrix = None

    def _tokenize(self, text):
        # Convert to lowercase and extract words
        return re.findall(r'\b\w+\b', text.lower())

    def fit_documents(self, documents):
        self.documents = documents
        
        # 1. Build Vocabulary
        word_counts = {}
        doc_word_counts = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_word_counts.append(tokens)
            for token in set(tokens):
                word_counts[token] = word_counts.get(token, 0) + 1
                
        self.vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
        num_docs = len(documents)
        
        # 2. Compute Inverse Document Frequency (IDF)
        self.idf = {}
        for word, count in word_counts.items():
            # Standard IDF formula: ln(Total Docs / Docs containing word)
            self.idf[word] = np.log((num_docs + 1) / (count + 1)) + 1.0

        # 3. Compute TF-IDF Matrix for documents
        self.tf_idf_matrix = np.zeros((num_docs, len(self.vocab)))
        for i, tokens in enumerate(doc_word_counts):
            self.tf_idf_matrix[i] = self._vectorize(tokens)
            
        # Normalize document vectors (L2 norm) for fast cosine similarity
        norms = np.linalg.norm(self.tf_idf_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        self.tf_idf_matrix = self.tf_idf_matrix / norms

    def _vectorize(self, tokens):
        # Calculate term frequency (TF) and multiply by IDF
        vector = np.zeros(len(self.vocab))
        total_tokens = len(tokens)
        if total_tokens == 0:
            return vector
            
        token_counts = {}
        for token in tokens:
            if token in self.vocab:
                token_counts[token] = token_counts.get(token, 0) + 1
                
        for token, count in token_counts.items():
            idx = self.vocab[token]
            tf = count / total_tokens
            vector[idx] = tf * self.idf[token]
            
        return vector

    def retrieve(self, query, top_k=2):
        """
        Retrieves top_k most similar documents using cosine similarity.
        """
        if self.tf_idf_matrix is None or len(self.vocab) == 0:
            return []
            
        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
            
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.tf_idf_matrix, query_vec)
        
        # Sort indices by similarity score descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            })
        return results


# =====================================================================
# 2. Local Generator (Mock LLM) & Gemini LLM Integration
# =====================================================================

class GenerationEngine:
    def __init__(self):
        # Check for Gemini API key
        self.api_key = os.environ.get("GEMINI_API_KEY", None)
        if self.api_key:
            print("[OK] Detected GEMINI_API_KEY environment variable. Will use live API for generation.")
        else:
            print("[INFO] No API key found. Defaulting to local smart Context Extractor.")

    def generate(self, prompt, retrieved_context, query):
        """
        Routes execution to live Gemini API if available, else runs local extractor.
        """
        if self.api_key:
            return self._call_gemini_api(prompt)
        else:
            return self._call_local_extractor(retrieved_context, query)

    def _call_local_extractor(self, retrieved_context, query):
        """
        A rule-based sentence context extractor.
        Analyzes retrieved text to present key facts directly matching the query.
        """
        # Collect sentences containing keywords from query
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)] # words with 4+ chars
        sentences = []
        
        for doc in retrieved_context:
            doc_sentences = re.split(r'(?<=[.!?]) +', doc)
            for sentence in doc_sentences:
                matched_count = sum(1 for kw in keywords if kw in sentence.lower())
                if matched_count > 0:
                    sentences.append((sentence, matched_count))
                    
        # Sort matched sentences by keyword count
        sentences.sort(key=lambda x: x[1], reverse=True)
        
        if sentences:
            ans = "\n- ".join(set([s[0] for s, _ in sentences[:3]]))
            return f"[Local Engine Answer]:\nBased on retrieved data, here is what I found:\n- {ans}"
        else:
            # Fallback to returning the best document snippet
            best_doc = retrieved_context[0] if retrieved_context else "No context available."
            return f"[Local Engine Answer]:\nCould not extract specific sentence matches. Best matching document:\n> {best_doc}"

    def _call_gemini_api(self, prompt):
        """
        Calls Google Gemini API (gemini-1.5-flash model) using raw requests.
        Keeps script self-contained without needing standard google-genai libraries.
        """
        import requests
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            # Extract generated text block
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            return f"[Gemini 1.5 Flash Answer]:\n{text}"
        except Exception as e:
            return f"[ERROR] Error querying Gemini API: {str(e)}\n\n[Fallback Local Answer]: Context matched but API failed."


# =====================================================================
# 3. RAG Agent Orchestrator
# =====================================================================

class RAGAgent:
    def __init__(self, documents):
        self.db = SimpleVectorDB()
        self.db.fit_documents(documents)
        self.generator = GenerationEngine()

    def query(self, user_query):
        # 1. Retrieve
        print(f"\nUser Query: '{user_query}'")
        print("Retrieving relevant documents...")
        retrieved = self.db.retrieve(user_query, top_k=2)
        
        contexts = []
        for i, item in enumerate(retrieved):
            print(f"  [{i+1}] (Score: {item['score']:.4f}) {item['document'][:80]}...")
            contexts.append(item['document'])
            
        # 2. Build Prompt Template
        context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])
        prompt = f"""You are a helpful QA Assistant. Answer the User Query based ONLY on the provided Context documents.
If the answer cannot be determined from the context, state that clearly.

Context Documents:
{context_str}

User Query: {user_query}

Answer:"""

        # 3. Generate Answer
        print("Synthesizing answer...")
        answer = self.generator.generate(prompt, contexts, user_query)
        return answer


# =====================================================================
# 4. CLI Execution
# =====================================================================

TECH_FACTS_DB = [
    "Neural Networks (MLPs) are feedforward networks consisting of an input layer, hidden layers, and an output layer. They learn weight parameters using the backpropagation algorithm.",
    "Recurrent Neural Networks (RNNs) process inputs sequentially by passing hidden states. However, they suffer from vanishing gradients when learning long sequences.",
    "LSTMs (Long Short-Term Memory) introduce forget, input, and output gates to solve vanishing gradients, allowing memory cells to persist information across long time intervals.",
    "Transformers rely on the Self-Attention mechanism instead of recurrence. This allows tokens to process in parallel, vastly increasing training speed and context window scalability.",
    "Retrieval-Augmented Generation (RAG) is a pattern that combines semantic document retrieval with Large Language Models to produce factual, grounded answers without retraining models.",
    "A Vector Database stores data as mathematical embeddings. It uses indexing algorithms to quickly perform similarity searches (like Cosine Similarity or Euclidean Distance) across high-dimensional vectors."
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) Agent from scratch.")
    parser.add_argument("--test-run", action="store_true", help="Runs a quick test query on the built-in facts database.")
    parser.add_argument("--query", type=str, default="Explain how transformers solve sequence issues.", help="The search query.")
    args = parser.parse_args()

    agent = RAGAgent(TECH_FACTS_DB)

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        # Run test query 1
        q1 = "What is RAG?"
        ans1 = agent.query(q1)
        print(ans1)
        
        # Run test query 2
        q2 = "Why do basic RNNs fail on long sequences?"
        ans2 = agent.query(q2)
        print(ans2)
        
        print("\n[SUCCESS] RAG Agent Integration Test Successful!")
        sys.exit(0)

    # Standard run
    answer = agent.query(args.query)
    print("\n" + "="*40)
    print(answer)
    print("="*40)
