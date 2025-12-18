# movie-plot-rag

# Mini RAG System for Movie Plots

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about movie plots using FAISS vector search and OpenAI's GPT.

## 🎯 What It Does

1. **Loads** a subset of the Wikipedia Movie Plots dataset
2. **Chunks** long plots into manageable pieces (~300 words)
3. **Embeds** chunks using OpenAI's embedding model
4. **Stores** vectors in a FAISS in-memory index
5. **Retrieves** top-k relevant chunks for any query
6. **Generates** structured answers with context and reasoning

## 📋 Requirements

- Python 3.8+
- OpenAI API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd movie-plot-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Or create it manually:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Important**: The `.env` file is already in `.gitignore` and won't be committed to git.

### 3. Download Dataset

Download the Wikipedia Movie Plots dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
- Place `wiki_movie_plots_deduped.csv` in the project directory

### 4. Run the System

```bash
python movie_rag.py wiki_movie_plots_deduped.csv
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```
pandas>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
openai>=1.0.0
```

## 🎬 Example Output

```json
{
  "answer": "Several movies feature artificial intelligence as a central theme. 'Alice in Wonderland' (1903) features early representations of fantastical elements. '2001: A Space Odyssey' notably features the HAL 9000 computer system that becomes antagonistic.",
  "contexts": [
    "Alice in Wonderland: Alice follows a white rabbit...",
    "The Great Train Robbery: Early action film...",
    "2001: A Space Odyssey: HAL 9000 computer..."
  ],
  "reasoning": "Retrieved 3 relevant plot excerpts. The answer was formed by analyzing mentions of the query topic across these movies: Alice in Wonderland, The Great Train Robbery, 2001: A Space Odyssey."
}
```

## 🏗️ System Architecture

```
┌─────────────┐
│   Dataset   │
│  (CSV file) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Chunk Plots │  (300 words/chunk)
│  & Embed    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ FAISS Index │  (In-memory vector store)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Query → Retrieve│  (Top-k chunks)
│ → Generate  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Structured │
│    Output   │
└─────────────┘
```

## 💡 Key Design Decisions

### Why These Technologies?

- **FAISS**: Fast, efficient, in-memory vector search (perfect for 200-500 movies)
- **OpenAI Embeddings**: High-quality semantic embeddings (text-embedding-3-small)
- **GPT-4o-mini**: Cost-effective, fast LLM for answer generation
- **No database**: Keeps system lightweight and portable

### Chunking Strategy

- 300 words per chunk balances context vs. precision
- Overlapping not needed for plot summaries (unlike technical docs)
- Preserves movie metadata with each chunk

### Retrieval Approach

- L2 distance in FAISS (cosine similarity alternative)
- Top-3 chunks provides sufficient context
- No reranking needed for this dataset size

## 🧪 Testing

The system runs three example queries on startup:
1. Movies featuring artificial intelligence
2. Movies with time travel
3. Romantic storylines in Paris

After examples, it enters interactive mode for custom queries.

## 🔧 Customization

Modify these parameters in `movie_rag.py`:

```python
# Sample size (line 53)
rag.load_and_preprocess(csv_path, sample_size=300)

# Chunk size (line 72)
max_words = 300

# Number of retrieved chunks (line 128)
k = 3

# Embedding model (line 26)
embedding_model = "text-embedding-3-small"

# LLM model (line 149)
model = "gpt-4o-mini"
```

## 📝 Project Structure

```
movie-plot-rag/
├── movie_rag.py           # Main RAG system
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── wiki_movie_plots_deduped.csv  # Dataset (download separately)
```

## 🎥 Video Walkthrough

[Link to 2-minute Loom video]

## ⏱️ Performance

- **Data loading**: ~2 seconds (300 movies)
- **Index building**: ~30-60 seconds (depends on API rate limits)
- **Query time**: ~2-3 seconds per query

## 🚨 Troubleshooting

**API Key Error**: Ensure `OPENAI_API_KEY` is set in your environment

**FAISS Import Error**: Install with `pip install faiss-cpu`

**Rate Limits**: The system processes embeddings sequentially. For larger datasets, implement batch processing.

**CSV Not Found**: Ensure the dataset path is correct

## 📄 License

MIT License - feel free to use and modify

## 🤝 Contributing

This is a take-home assignment implementation. Feedback welcome!

---

Built with ❤️ for the RAG take-home challenge