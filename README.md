# Medium Article RAG Assistant

This project builds a Retrieval-Augmented Generation assistant over the provided Medium articles dataset.

## Current RAG Settings

The assignment-facing stats endpoint reports:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.1,
  "top_k": 7
}
```

`chunk_size = 512` is the target chunk size in approximate tokens. The current sample chunking script implements this as `350` words, because 350 English words is roughly 512 tokens. The overlap is `35` words, which is approximately `0.1` of the chunk size.

For UTF-8-safe local test output on Windows, prefer running Python checks with:

```powershell
py -X utf8 ...
```
