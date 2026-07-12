# Project 4: Retrieval-Augmented Generation (RAG) Agent

A production-style Retrieval-Augmented Generation (RAG) agent that parses data, stores semantic representations in a local vector database, retrieves relevant contexts, and orchestrates prompt routing to synthesize verified, grounded answers.

This project implements a complete **TF-IDF Vector Database** from scratch using only Python and NumPy, showing the mathematical alignment of classic Information Retrieval (IR) and modern Generative AI.

---

## 📐 Systems Architecture

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

## 📐 Mathematical Foundations

### 1. TF-IDF Representation (Term Frequency - Inverse Document Frequency)
TF-IDF calculates the importance of a term $t$ in a document $d$ relative to a corpus of documents $D$.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Term Frequency ($\text{TF}(t, d)$)**: The raw count of term $t$ in document $d$ normalized by the total tokens in $d$.
  $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$$
- **Inverse Document Frequency ($\text{IDF}(t, D)$)**: Measures how much information the term provides (whether it's common or rare across all documents). We use the smoothed natural logarithm formula:
  $$\text{IDF}(t, D) = \ln\left(\frac{N + 1}{DF(t) + 1}\right) + 1$$
  *Where $N$ is the total documents in corpus $D$, and $DF(t)$ is the number of documents containing term $t$.*

### 2. Cosine Similarity Scoring
To rank documents, we measure the cosine angle between the query vector $\mathbf{q}$ and document vectors $\mathbf{d}$. The closer the vectors are in high-dimensional space, the higher the score:

$$\text{Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_i q_i d_i}{\sqrt{\sum_i q_i^2} \sqrt{\sum_i d_i^2}}$$

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

## 💡 Key Takeaways for Recruiters
- **Zero-Dependency Vector DB**: Implements document parsing, tokenization, vocabulary generation, TF-IDF calculation, and cosine ranking using only NumPy.
- **Smart Generation Routing**: Supports a dual generation pipeline. If no API key is set, it falls back to a smart context matching algorithm that extracts key facts to avoid empty responses.
- **API Call Optimization**: Calls the Gemini API using native Python requests (handling JSON payloads and extracting candidate contents), avoiding heavy framework overhead.
