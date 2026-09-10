# Portfolio Website

A multi-page personal portfolio: **Home**, **Projects** (a scrollable picker), **Certificates**, and **About**, plus a project detail page ("Neurons to Transformers") with four real, live, in-browser ML demos running on actual trained model weights — no mocked data anywhere.

## Running it locally

This is a plain static site (no build step, no framework) — but it **must be served over `http://`, not opened directly as a `file://` URL**. The live demos `fetch()` JSON weight files from `model_weights/`, and browsers block `fetch()` of local files under the `file://` protocol. Double-clicking `index.html` will load the page but the demos will silently fail to load their weights.

From this `portfolio-website/` directory, run a local server:

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/index.html** in your browser.

Any static file server works instead — e.g. VS Code's "Live Server" extension, or `npx serve`.

## Pages

| File | What it is |
|---|---|
| `index.html` | Home — intro, quick links, featured project |
| `projects.html` | Projects picker (scroll/drag/arrow-keys, tick sound, snap-to-center) |
| `neurons-to-transformers.html` | Project detail page: architecture overview + 4 live demos (MLP, RNN/LSTM, Transformer attention, RAG) |
| `certificates.html` | Searchable/filterable grid of real certificates (PDFs in `certificates/`) |
| `about.html` | Bio, experience timeline, skills |

Shared across all pages: `style.css` (design tokens + all component styles) and `common.js` (nav active-state highlighting). Each other page loads only the JS it actually needs — `neurons-to-transformers.js`, `certificates.js`, or `projects-picker.js` — so the ~1-2MB of model weights only load on the one page that uses them.

## The live demos

`neurons-to-transformers.js` re-implements each model's real forward pass by hand in JavaScript (matrix ops, LSTM gates, causal self-attention, TF-IDF cosine similarity) and runs it on weights exported straight from the trained PyTorch/NumPy models in `model_weights/*.json`. If you retrain a project, regenerate its weights with the matching script from the repo root:

```bash
python scripts/export_mlp_weights.py
python scripts/export_rnn_weights.py
python scripts/export_lstm_weights.py
python scripts/export_transformer_weights.py
python scripts/export_rag_facts.py
```

Each writes its output straight into `portfolio-website/model_weights/`.

## Adding a new project to the picker

Edit the `PROJECTS` array at the top of `projects-picker.js` — add `{ id, title, tagline, tags, href, githubHref }`. No other wiring needed; the reel, preview panel, and keyboard/scroll handling are all data-driven.
