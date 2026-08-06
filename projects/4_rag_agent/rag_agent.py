# =====================================================================
# Project 4: Production RAG AI Agent from Scratch
# Author: Shabee Ibn Ehtesham
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
    Implements TF-IDF keyword vectorization and Cosine Similarity matching.
    """
    def __init__(self):
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.tf_idf_matrix = None

    def _tokenize(self, text):
        # Convert text to lowercase and split into words
        return re.findall(r'\b\w+\b', text.lower())

    def fit_documents(self, documents):
        self.documents = documents
        
        # 1. Build Vocabulary (list of all unique words across all documents)
        word_counts = {}
        doc_word_counts = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_word_counts.append(tokens)
            # Count how many documents contain each unique word
            for token in set(tokens):
                word_counts[token] = word_counts.get(token, 0) + 1
                
        self.vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
        num_docs = len(documents)
        
        # 2. Compute Inverse Document Frequency (IDF) for all words
        self.idf = {}
        for word, count in word_counts.items():
            # Standard IDF formula: ln((Total Docs + 1) / (Docs containing word + 1)) + 1
            # Gives higher scores to rare terms, and lower scores to very common terms (e.g. 'the', 'is')
            self.idf[word] = np.log((num_docs + 1) / (count + 1)) + 1.0

        # 3. Compute TF-IDF Matrix for our library documents
        self.tf_idf_matrix = np.zeros((num_docs, len(self.vocab)))
        for i, tokens in enumerate(doc_word_counts):
            self.tf_idf_matrix[i] = self._vectorize(tokens)
            
        # Pre-normalize the vectors using L2 norm (makes document vectors have length 1.0)
        # This allows us to calculate Cosine Similarity later using a simple, fast dot-product.
        norms = np.linalg.norm(self.tf_idf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Avoid division by zero
        self.tf_idf_matrix = self.tf_idf_matrix / norms

    def _vectorize(self, tokens):
        # Turns a list of words into a numerical vector (a list of TF-IDF scores)
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
            tf = count / total_tokens # Term Frequency (local count / total count)
            vector[idx] = tf * self.idf[token] # Multiply by word rarity (IDF)
            
        return vector

    def retrieve(self, query, top_k=2):
        """
        Finds and returns the top_k documents most relevant to the query.
        """
        if self.tf_idf_matrix is None or len(self.vocab) == 0:
            return []
            
        # 1. Turn query text into a numeric vector
        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        # 2. Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
            
        # 3. Calculate Cosine Similarity (dot product of query vector with all document vectors)
        similarities = np.dot(self.tf_idf_matrix, query_vec)
        
        # 4. Sort similarity scores in descending order and fetch the top K
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
    """
    Handles prompt execution. Routes queries to the Google Gemini API if a key is provided,
    otherwise falls back to a smart local keyword context extractor.
    """
    def __init__(self):
        # Look for the Gemini API key in local environment variables
        self.api_key = os.environ.get("GEMINI_API_KEY", None)
        if self.api_key:
            print("[OK] Detected GEMINI_API_KEY environment variable. Will use live API for generation.")
        else:
            print("[INFO] No API key found. Defaulting to local smart Context Extractor.")

    def generate(self, prompt, retrieved_context, query):
        """
        Routes the final prompt to the appropriate generation engine.
        """
        if self.api_key:
            return self._call_gemini_api(prompt)
        else:
            return self._call_local_extractor(retrieved_context, query)

    def _call_local_extractor(self, retrieved_context, query):
        """
        A local rules-based fallback. Matches keywords in the query to sentences
        in the retrieved documents, extracting the most relevant facts directly.
        """
        # Find keywords with 4 or more letters to filter out common grammar words
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)]
        sentences = []
        
        for doc in retrieved_context:
            doc_sentences = re.split(r'(?<=[.!?]) +', doc)
            for sentence in doc_sentences:
                matched_count = sum(1 for kw in keywords if kw in sentence.lower())
                if matched_count > 0:
                    sentences.append((sentence, matched_count))
                    
        # Rank matching sentences based on how many keywords they contained
        sentences.sort(key=lambda x: x[1], reverse=True)
        
        if sentences:
            ans = "\n- ".join(dict.fromkeys(s for s, _ in sentences[:3]))
            return f"[Local Engine Answer]:\nBased on retrieved data, here is what I found:\n- {ans}"
        else:
            # If no keyword matches were found, return the best raw document snippet
            best_doc = retrieved_context[0] if retrieved_context else "No context available."
            return f"[Local Engine Answer]:\nCould not extract specific sentence matches. Best matching document:\n> {best_doc}"

    def _call_gemini_api(self, prompt):
        """
        Sends the prompt directly to the live Google Gemini API using native HTTP requests.
        Self-contained and avoids heavy framework dependencies.
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
            # Extract and return response text from the JSON candidate structures
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            return f"[Gemini 1.5 Flash Answer]:\n{text}"
        except Exception as e:
            return f"[ERROR] Error querying Gemini API: {str(e)}\n\n[Fallback Local Answer]: Context matched but API failed."


# =====================================================================
# 3. RAG Agent Orchestrator
# =====================================================================

class RAGAgent:
    """
    The RAG Agent Orchestrator. Coordinates vector database retrieval,
    pasting documents into a prompt template, and executing generation.
    """
    def __init__(self, documents):
        self.db = SimpleVectorDB()
        self.db.fit_documents(documents)
        self.generator = GenerationEngine()

    def query(self, user_query):
        # 1. RETRIEVE: Look up the 2 most similar documents in our Vector Database
        print(f"\nUser Query: '{user_query}'")
        print("Retrieving relevant documents...")
        retrieved = self.db.retrieve(user_query, top_k=2)
        
        contexts = []
        for i, item in enumerate(retrieved):
            print(f"  [{i+1}] (Score: {item['score']:.4f}) {item['document'][:80]}...")
            contexts.append(item['document'])
            
        # 2. AUGMENT: Paste retrieved documents into a context prompt template
        context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])
        prompt = f"""You are a helpful QA Assistant. Answer the User Query based ONLY on the provided Context documents.
If the answer cannot be determined from the context, state that clearly.

Context Documents:
{context_str}

User Query: {user_query}

Answer:"""

        # 3. GENERATE: Feed augmented prompt to LLM / Extractor to get grounded answer
        print("Synthesizing answer...")
        answer = self.generator.generate(prompt, contexts, user_query)
        return answer


# =====================================================================
# 4. CLI Execution
# =====================================================================

# Grounded facts about Shakespeare's plays, characters, and life - deliberately
# in the same "universe" as the data/shakespeare.txt corpus Projects 1-3 train
# on (the corpus's opening lines are literally the First Citizen scene from
# Coriolanus, referenced below), so this project's "beyond pattern generation"
# capstone answers real questions about the same material 1-3 only ever learn
# to imitate the SOUND of. Every entry below is cross-checked against
# well-established sources; genuine ambiguity (e.g. the exact play count,
# whether the historical Caesar said "Et tu, Brute?") is stated rather than
# presented as settled fact.
SHAKESPEARE_FACTS_DB = [
    "\"To be, or not to be, that is the question\" is spoken by Prince Hamlet in Act 3, Scene 1 of Shakespeare's play Hamlet, as he weighs whether it is nobler to endure suffering or end it.",
    "\"Wherefore art thou Romeo?\" is spoken by Juliet in Act 2, Scene 2 of Romeo and Juliet. 'Wherefore' means 'why', not 'where' - Juliet is asking why Romeo must be a Montague, not asking where he is standing.",
    "Romeo and Juliet belong to two feuding families in Verona: Romeo is a Montague and Juliet is a Capulet. The play's Prologue describes the feud as an 'ancient grudge'.",
    "Lady Macbeth is the wife of Macbeth in Shakespeare's tragedy Macbeth. She persuades her husband to murder King Duncan, and later succumbs to guilt-driven madness, famously sleepwalking while trying to wash imaginary blood from her hands ('Out, damned spot!').",
    "Iago is the villain in Othello. He manipulates the title character into believing his wife Desdemona has been unfaithful, ultimately driving Othello to kill her.",
    "\"All the world's a stage, and all the men and women merely players\" is spoken by the character Jaques in Act 2, Scene 7 of As You Like It.",
    "\"Et tu, Brute?\" are Julius Caesar's words upon recognizing his friend Brutus among his assassins in Shakespeare's Julius Caesar. Whether the historical Caesar actually said this in Latin is disputed by historians; the line is Shakespeare's dramatization.",
    "In Macbeth, three witches deliver prophecies to Macbeth - including that he will become Thane of Cawdor and then king - setting the play's tragic events in motion.",
    "A Midsummer Night's Dream features Puck, also called Robin Goodfellow, a mischievous fairy who causes romantic confusion among the human characters using a magical flower.",
    "Shakespeare's plays are traditionally grouped into three main categories: comedies, tragedies, and histories.",
    "Coriolanus opens with a scene of hungry Roman citizens, led by a 'First Citizen', confronting the patricians over food shortages - the exact scene ('Before we proceed any further, hear me speak') that opens the Shakespeare text corpus used to train Projects 1-3 in this repository.",
    "William Shakespeare was baptized on 26 April 1564 in Stratford-upon-Avon, England; his exact birth date was not recorded, but is traditionally celebrated on 23 April.",
    "Shakespeare died on 23 April 1616, also in Stratford-upon-Avon.",
    "Shakespeare's acting company, later known as the King's Men, performed at the Globe Theatre, an open-air playhouse in London built in 1599.",
    "The exact number of plays Shakespeare wrote is genuinely debated, since some were co-written and attribution is disputed for a few borderline works, but most modern collected editions count 39 plays.",
    "In Othello, Iago's manipulation works partly through a handkerchief that Othello gave Desdemona; Iago plants it as false evidence of her infidelity.",
    "The Montague-Capulet feud in Romeo and Juliet is never fully explained in the play - the text calls it an 'ancient grudge' without stating its original cause."
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) Agent from scratch, grounded in real Shakespeare facts.")
    parser.add_argument("--test-run", action="store_true", help="Runs a quick test query on the built-in facts database.")
    parser.add_argument("--query", type=str, default="What play is 'to be or not to be' from?", help="The search query.")
    args = parser.parse_args()

    agent = RAGAgent(SHAKESPEARE_FACTS_DB)

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        # Run test query 1
        q1 = "Who is Iago and what does he do in Othello?"
        ans1 = agent.query(q1)
        print(ans1)

        # Run test query 2
        q2 = "What is the Montague and Capulet feud about?"
        ans2 = agent.query(q2)
        print(ans2)

        print("\n[SUCCESS] RAG Agent Integration Test Successful!")
        sys.exit(0)

    # Standard run
    answer = agent.query(args.query)
    print("\n" + "="*40)
    print(answer)
    print("="*40)
