# Medium RAG Assistant

A live Retrieval-Augmented Generation assistant built over a Medium articles dataset.

Ask it a question, and it does the RAG dance:

1. embeds the question,
2. retrieves relevant Medium article chunks,
3. builds an augmented prompt,
4. answers only from the retrieved article context.

If the answer is not supported by the retrieved Medium articles, it should say:

```text
I don't know based on the provided Medium articles data.
```

## Live Project

```text
Live URL:
https://medium-rag-assistant-alpha.vercel.app

GitHub:
https://github.com/DimaBsharaa/medium-rag-assistant
```

## API Endpoints

### Health Check

```http
GET /
```

Returns a small message showing that the API is running.

### RAG Prompt

```http
POST /api/prompt
```

Input:

```json
{
  "question": "Your question here"
}
```

Output:

```json
{
  "response": "Final answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Article title",
      "chunk": "Retrieved article chunk",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt used for the chat model",
    "User": "User prompt with the question and retrieved context"
  }
}
```

### Stats

```http
GET /api/stats
```

Returns the current RAG configuration:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.1,
  "top_k": 7
}
```

## Dataset

This project uses the Medium articles CSV dataset provided for the assignment.

The original CSV is not committed to GitHub. The deployed app includes only the processed runtime lookup file needed for hybrid retrieval:

```text
data/chunks_full.csv
```

The final processed dataset contains:

```text
7,682 articles
29,324 chunks
```

## Models And Services

LLMod.ai course API:

```text
Embedding model:
4UHRUIN-text-embedding-3-small

Chat model:
4UHRUIN-gpt-5-mini
```

Vector database:

```text
Pinecone
1536-dimensional vectors
cosine similarity
```

## How Retrieval Works

The project started simple: chunk articles, embed them, upload vectors to Pinecone, and retrieve with semantic search.

During testing, one difficult question showed that vector search alone could miss rare but important words. So the final retriever uses a hybrid approach:

```text
Pinecone vector search
+
local keyword search over article chunks
+
deduplication and context selection
```

This helps with both semantic questions and precise wording questions, such as questions involving article titles, rare terms, or specific concepts.

For list-style questions, the API keeps at most 3 unique articles. For normal questions, it keeps the context compact so the chat model sees the strongest passages.

## Chunking

The assignment-facing target is:

```text
chunk_size = 512
overlap_ratio = 0.1
top_k = 7
```

The chunking script implements the 512-token target approximately as:

```text
350 words per chunk
35 words overlap
```

That keeps chunks large enough to preserve article meaning, but small enough to retrieve and pass into the model cleanly.

## Project Structure

```text
api/
  index.py                 Vercel entry point

app/
  main.py                  FastAPI routes
  rag.py                   RAG pipeline: retrieval, prompts, chat call, response formatting

scripts/
  01_read_data.py          Dataset sanity check
  02_chunk_articles.py     Chunking logic
  02b_chunk_full_articles.py
  03_embed_chunks.py       Embedding logic
  03b_embed_full_chunks.py
  04_upload_to_pinecone.py
  04b_upload_full_to_pinecone.py
  05_search_pinecone.py
  06_answer_question.py    Local RAG answer script

data/
  chunks_full.csv          Runtime lookup file for deployed hybrid retrieval
```

## Environment Variables

The app expects these variables in local `.env` and in Vercel project settings:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.llmod.ai
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=medium-rag-assistant
```

Secrets and raw/embedded data are ignored by Git.

## What It Can Answer

The assistant was validated live on questions that ask it to:

- find one precise article and return the title and author,
- list exactly three articles about a topic,
- identify an article's main idea and summarize it,
- recommend an article with evidence from retrieved text,
- refuse outside-knowledge questions when the answer is not in the retrieved Medium context.

Some examples of the kinds of questions it handled:

```text
Find an article about writers who dislike self-promotion.
List exactly three articles about education.
Find an article connecting past pandemics with innovation and recovery.
Recommend practical advice for building habits that stick.
Explain why writers should draft more than one headline.
```

## Notes

Development started with a small 100-article sample to avoid unnecessary embedding cost. After the local pipeline worked end-to-end, the full dataset was chunked, embedded once, and uploaded to Pinecone.

The final system is not just a prompt wrapped around a model. It is a full RAG pipeline: dataset processing, embeddings, Pinecone storage, hybrid retrieval, strict context-grounded prompting, FastAPI endpoints, and a live Vercel deployment.
