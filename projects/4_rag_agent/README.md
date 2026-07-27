# Project 4: Retrieval-Augmented Generation (RAG) Agent

A production-style Retrieval-Augmented Generation (RAG) agent that parses data, stores semantic representations in a local vector database, retrieves relevant contexts, and orchestrates prompt routing to synthesize verified, grounded answers.

This project implements a complete **TF-IDF Vector Database** from scratch using only Python and NumPy, showing the mathematical alignment of classic Information Retrieval (IR) and modern Generative AI.

---

## ⚡ Systems Architecture

```
                  ┌──────────────────┐
                  │    User Query    │
                  └────────┬─────────┘
                           ▼
            ┌────────────────────────────┐
            │   Vector Database Search   │
            │   (TF-IDF Cosine Sim)      │
            └─────────────┬──────────────┘
                           ▼
                Top-K Context Retrieval
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   [Context Found]                 [No Context Found]
           │                               │
           ▼                               ▼
    Synthesize Prompt               Direct Fallback/
   w/ Context Docs                  Grounding check
           │                               │
           └───────────────┬───────────────┘
                           ▼
              ┌────────────────────────┐
              │   Generation Engine    │
              │   (Local or Gemini API)│
              └────────────┬───────────┘
                           ▼
                     Final Answer
```

---

## 💡 The Intuitive "Library Filing Cards" Analogy

Large Language Models (LLMs) are like smart, well-read assistants. However, they only know things they read during training. If you ask them about your private documents, they won't know the answer. 

**Retrieval-Augmented Generation (RAG)** solves this by looking up the correct facts in a database and pasting them into the prompt before sending it to the LLM. 

Here is how our custom Vector Database finds the right documents, using a **Library Index Card** analogy:

1. **The Catalog Cards (Document Collection)**: Every text block in our database is like a book in a library. To find books quickly, we build a cabinet of "index cards" describing them.
2. **TF (Term Frequency - Local Importance)**: How many times a word is written on a specific index card. If the word "conveyor" appears 10 times in a small article, then "conveyor" is very important to that specific article.
3. **IDF (Inverse Document Frequency - Global Rarity)**: How rare a word is across the *whole library*. 
   * Common words like "the", "is", or "and" appear on almost every card, so they have a **low IDF** (they don't tell us much about the book's topic).
   * Rare words like "LSTM", "Xavier", or "TF-IDF" only appear on a few cards, so they have a **high IDF** (they are highly descriptive keywords!).
4. **Vector Search (Filing Card Matching)**:
   * When you type a query (e.g., "Why do basic RNNs fail?"), we make a "query index card" with TF-IDF scores for your keywords.
   * **Cosine Similarity** compares your query card to every document card. It measures the overlap of rare keywords. The cards that have the highest score (most aligned keywords) are retrieved and handed to the generation engine!

---

## 📐 Mathematical Foundations

### 1. TF-IDF Representation (Term Frequency - Inverse Document Frequency)
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

* **Term Frequency ($\text{TF}(t, d)$)**: The raw count of term $t$ in document $d$ normalized by total words.
* **Inverse Document Frequency ($\text{IDF}(t, D)$)**: Measures keyword rarity.
  $$\text{IDF}(t, D) = \ln\left(\frac{N + 1}{DF(t) + 1}\right) + 1$$
  *Where $N$ is total documents, and $DF(t)$ is number of documents containing term $t$.*

### 2. Cosine Similarity Scoring
To rank documents, we measure the cosine angle between the query vector $\mathbf{q}$ and document vectors $\mathbf{d}$:
$$\text{Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

In our database, we pre-normalize all document vectors ($\|\mathbf{d}\|_2 = 1.0$). Therefore, query matching simplifies to a single fast dot-product operation: $\mathbf{q} \cdot \mathbf{d}$.

---

## 🚀 How to Run

### Integration Test
Run a quick test query on the built-in tech facts database:
```bash
python rag_agent.py --test-run
```

### Run Custom Queries
Run a custom query using the local keyword-matching generation engine:
```bash
python rag_agent.py --query "Why do basic RNNs fail on long sequences?"
```

### Run with Live Gemini API
To run the generator with the state-of-the-art **Gemini 1.5 Flash** model:
1. Obtain a free Gemini API Key from Google AI Studio.
2. Set the environment variable:
   ```powershell
   # In PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
3. Run the query script:
   ```powershell
   python rag_agent.py --query "Summarize the differences between RNNs and Transformers based on the documents."
   ```

---

## 🧪 Technical Notes

A few design decisions I thought carefully about:

- **Building the Vector DB from scratch**: I deliberately avoided using FAISS or ChromaDB — I wanted to understand exactly how TF-IDF scoring and cosine similarity work at the numpy level. Pre-normalizing all document vectors at index time means retrieval is just a dot product, which is fast.
- **Fallback generation without an API key**: Rather than crashing or returning nothing when no API key is set, I built a local keyword-matching fallback that extracts and stitches together the most relevant sentences from retrieved documents. Not as fluent as an LLM, but it proves the retrieval layer works correctly on its own.
- **Calling the Gemini API with raw `requests`**: I wanted to avoid pulling in the entire `google-generativeai` SDK for what is essentially one POST request. Parsing the JSON response manually keeps the dependencies minimal.
