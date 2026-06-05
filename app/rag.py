import math
import re
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

# Reported assignment hyperparameter: the sample chunking script approximates
# this 512-token target with 350-word chunks for English text.
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.1
TOP_K = 7
KEYWORD_CANDIDATES = 20
MAX_MULTI_RESULT_ARTICLES = 3
MAX_CONTEXT_CHUNKS_FOR_CHAT = 5
MAX_CHUNKS_PER_ARTICLE = 2
CONTENT_FILTER_RETRY_CHUNKS = 3
MAX_COMPLETION_TOKENS = 1000
CHUNKS_LOOKUP_PATH = Path("data/chunks_full.csv")
MULTI_RESULT_PATTERNS = (
    r"\blist\b",
    r"\bexactly\s+3\b",
    r"\b3\s+articles\b",
    r"\bthree\s+articles\b",
    r"\bmultiple\s+(?:articles|results|titles)\b",
)
CONTENT_FILTER_MARKERS = (
    "ContentPolicyViolation",
    "ResponsibleAIPolicyViolation",
    "content_filter",
    "content management policy",
)
STOPWORDS = {
    "about",
    "above",
    "actually",
    "after",
    "again",
    "against",
    "aimed",
    "also",
    "and",
    "article",
    "articles",
    "author",
    "based",
    "because",
    "before",
    "being",
    "better",
    "between",
    "can",
    "cannot",
    "central",
    "could",
    "exactly",
    "find",
    "from",
    "give",
    "have",
    "into",
    "its",
    "list",
    "main",
    "more",
    "only",
    "provide",
    "question",
    "recommend",
    "recommended",
    "recommender",
    "return",
    "should",
    "such",
    "summarise",
    "summarize",
    "summary",
    "than",
    "the",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "three",
    "title",
    "titles",
    "want",
    "what",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""


load_dotenv()
_CHUNKS_CACHE = None


class RagPipelineError(RuntimeError):
    def __init__(
        self,
        step,
        original_error,
        context_count=None,
        context_summary=None,
    ):
        self.step = step
        self.original_error = original_error
        self.context_count = context_count
        self.context_summary = context_summary or []
        super().__init__(
            f"RAG pipeline failed during {step}: "
            f"{type(original_error).__name__}: {original_error}"
        )


class HybridMatch:
    def __init__(
        self,
        score,
        metadata,
        sources=None,
        keyword_score=0.0,
        matched_terms=None,
        phrase_hits=None,
    ):
        self.score = float(score)
        self.metadata = metadata
        self.sources = sources or []
        self.keyword_score = float(keyword_score)
        self.matched_terms = matched_terms or []
        self.phrase_hits = phrase_hits or []


def require_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def validate_environment():
    require_env_var("OPENAI_API_KEY")
    require_env_var("OPENAI_BASE_URL")
    require_env_var("PINECONE_API_KEY")
    require_env_var("PINECONE_INDEX_NAME")


def build_embeddings_client():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_chat_client():
    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).lower()


def tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.lower())


def extract_query_terms(question):
    terms = []
    for token in tokenize(question):
        if token not in STOPWORDS and not token.isdigit():
            terms.append(token)

    return list(dict.fromkeys(terms))


def extract_query_phrases(terms):
    phrases = []
    for size in (3, 2):
        for start in range(0, len(terms) - size + 1):
            phrases.append(" ".join(terms[start : start + size]))

    return phrases


def load_chunks_cache():
    global _CHUNKS_CACHE

    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    if not CHUNKS_LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Missing hybrid retrieval lookup file: {CHUNKS_LOOKUP_PATH}"
        )

    chunks_df = pd.read_csv(CHUNKS_LOOKUP_PATH)
    required_columns = [
        "article_id",
        "title",
        "authors",
        "url",
        "timestamp",
        "tags",
        "chunk_index",
        "chunk_text",
    ]
    for column in required_columns:
        if column not in chunks_df.columns:
            raise ValueError(
                f"Missing required column in {CHUNKS_LOOKUP_PATH}: {column}"
            )

    chunks = []
    for _, row in chunks_df.iterrows():
        metadata = {
            "article_id": str(row["article_id"]),
            "title": row["title"],
            "authors": row["authors"],
            "url": row["url"],
            "timestamp": row["timestamp"],
            "tags": row["tags"],
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": row["chunk_text"],
        }
        search_text = " ".join(
            [
                str(metadata["title"]),
                str(metadata["tags"]),
                str(metadata["chunk_text"]),
            ]
        ).lower()
        chunks.append(
            {
                "id": f"{metadata['article_id']}_{metadata['chunk_index']}",
                "metadata": metadata,
                "search_text": search_text,
            }
        )

    _CHUNKS_CACHE = chunks
    return _CHUNKS_CACHE


def compute_document_frequency(chunks, terms):
    total_docs = len(chunks)
    idf = {}

    for term in terms:
        doc_frequency = sum(1 for chunk in chunks if term in chunk["search_text"])
        idf[term] = math.log((total_docs + 1) / (doc_frequency + 1)) + 1

    return idf


def keyword_search(question):
    chunks = load_chunks_cache()
    terms = extract_query_terms(question)
    phrases = extract_query_phrases(terms)
    idf = compute_document_frequency(chunks, terms)
    min_term_matches = 1 if len(terms) <= 1 else 2
    scored = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        search_text = chunk["search_text"]
        matched_terms = []
        score = 0.0

        for term in terms:
            term_count = search_text.count(term)
            if term_count:
                matched_terms.append(term)
                score += min(term_count, 3) * idf.get(term, 1.0)

        phrase_hits = []
        for phrase in phrases:
            if phrase and phrase in search_text:
                phrase_hits.append(phrase)
                phrase_size = len(phrase.split())
                phrase_idf = sum(idf.get(term, 1.0) for term in phrase.split())
                score += 12.0 * phrase_size + phrase_idf

        if not matched_terms and not phrase_hits:
            continue

        if len(set(matched_terms)) < min_term_matches and not phrase_hits:
            continue

        title_text = normalize_text(metadata.get("title"))
        tags_text = normalize_text(metadata.get("tags"))
        title_hits = [term for term in terms if term in title_text]
        tag_hits = [term for term in terms if term in tags_text]
        unique_matches = set(matched_terms)

        score += 5.0 * len(title_hits)
        score += 3.0 * len(tag_hits)
        score += 3.0 * (len(unique_matches) ** 2)

        if phrase_hits:
            score += 10.0 * len(phrase_hits)

        scored.append(
            {
                "id": chunk["id"],
                "metadata": metadata,
                "keyword_score": score,
                "matched_terms": matched_terms,
                "phrase_hits": phrase_hits,
            }
        )

    return sorted(
        scored,
        key=lambda item: item["keyword_score"],
        reverse=True,
    )[:KEYWORD_CANDIDATES]


def vector_search(question):
    embeddings_client = build_embeddings_client()
    query_vector = embeddings_client.embed_query(question)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    matches = []
    for match in results.matches:
        metadata = match.metadata or {}
        article_id = str(metadata.get("article_id"))
        chunk_index = int(metadata.get("chunk_index"))
        matches.append(
            {
                "id": f"{article_id}_{chunk_index}",
                "metadata": {
                    "article_id": article_id,
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors"),
                    "url": metadata.get("url"),
                    "timestamp": metadata.get("timestamp"),
                    "tags": metadata.get("tags"),
                    "chunk_index": chunk_index,
                    "chunk_text": metadata.get("chunk_text"),
                },
                "vector_score": float(match.score),
            }
        )

    return matches


def merge_hybrid_results(vector_matches, keyword_matches):
    merged = {}

    for match in vector_matches:
        item = merged.setdefault(
            match["id"],
            {
                "metadata": match["metadata"],
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "matched_terms": [],
                "phrase_hits": [],
                "sources": [],
            },
        )
        item["vector_score"] = match["vector_score"]
        item["metadata"] = match["metadata"]
        item["sources"].append("vector")

    for match in keyword_matches:
        item = merged.setdefault(
            match["id"],
            {
                "metadata": match["metadata"],
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "matched_terms": [],
                "phrase_hits": [],
                "sources": [],
            },
        )
        item["keyword_score"] = match["keyword_score"]
        item["metadata"] = match["metadata"]
        item["matched_terms"] = match["matched_terms"]
        item["phrase_hits"] = match["phrase_hits"]
        if "keyword" not in item["sources"]:
            item["sources"].append("keyword")

    max_keyword_score = max(
        [item["keyword_score"] for item in merged.values()] or [1.0]
    )
    matches = []

    for item in merged.values():
        normalized_keyword = item["keyword_score"] / max_keyword_score
        hybrid_score = item["vector_score"] + normalized_keyword
        matches.append(
            HybridMatch(
                score=hybrid_score,
                metadata=item["metadata"],
                sources=item["sources"],
                keyword_score=item["keyword_score"],
                matched_terms=item["matched_terms"],
                phrase_hits=item["phrase_hits"],
            )
        )

    return sorted(matches, key=lambda match: match.score, reverse=True)


def query_hybrid(question):
    vector_matches = vector_search(question)
    keyword_matches = keyword_search(question)
    return merge_hybrid_results(vector_matches, keyword_matches)


def is_multi_result_question(question):
    question_lower = question.lower()
    return any(
        re.search(pattern, question_lower) for pattern in MULTI_RESULT_PATTERNS
    )


def dedupe_matches(matches):
    seen = set()
    deduped = []

    for match in matches:
        metadata = match.metadata or {}
        key = metadata.get("article_id") or metadata.get("title")

        if key in seen:
            continue

        seen.add(key)
        deduped.append(match)

    return deduped


def select_context_matches(question, matches):
    if is_multi_result_question(question):
        return dedupe_matches(matches)[:MAX_MULTI_RESULT_ARTICLES]

    per_article_counts = Counter()
    selected = []

    for match in matches:
        metadata = match.metadata or {}
        article_id = metadata.get("article_id")
        sources = getattr(match, "sources", [])
        matched_terms = getattr(match, "matched_terms", [])
        phrase_hits = getattr(match, "phrase_hits", [])

        if sources == ["keyword"]:
            if not phrase_hits and len(set(matched_terms)) < 4:
                continue

        if per_article_counts[article_id] >= MAX_CHUNKS_PER_ARTICLE:
            continue

        per_article_counts[article_id] += 1
        selected.append(match)

        if len(selected) >= MAX_CONTEXT_CHUNKS_FOR_CHAT:
            break

    return selected


def build_context_string(matches):
    context_parts = []

    for rank, match in enumerate(matches, start=1):
        metadata = match.metadata or {}
        context_parts.append(
            "\n".join(
                [
                    f"[Context {rank}]",
                    f"score: {match.score}",
                    f"article_id: {metadata.get('article_id')}",
                    f"title: {metadata.get('title')}",
                    f"authors: {metadata.get('authors')}",
                    f"chunk_index: {metadata.get('chunk_index')}",
                    f"chunk_text: {metadata.get('chunk_text')}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_user_prompt(question, context):
    return f"""User question:
{question}

Retrieved Medium article context:
{context}

Answer the user question using only the retrieved context above."""


def build_context_response(matches):
    context = []

    for match in matches:
        metadata = match.metadata or {}
        context.append(
            {
                "article_id": str(metadata.get("article_id")),
                "title": metadata.get("title"),
                "chunk": metadata.get("chunk_text"),
                "score": float(match.score),
            }
        )

    return context


def build_context_summary(matches):
    summary = []

    for match in matches:
        metadata = match.metadata or {}
        summary.append(
            {
                "article_id": str(metadata.get("article_id")),
                "title": metadata.get("title"),
                "chunk_index": metadata.get("chunk_index"),
                "score": float(match.score),
            }
        )

    return summary


def is_content_filter_error(error):
    error_text = str(error)
    return any(marker in error_text for marker in CONTENT_FILTER_MARKERS)


def invoke_chat(user_prompt):
    chat_client = build_chat_client()
    return chat_client.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    )


def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])

        return "".join(text_parts)

    return str(content) if content is not None else ""


def build_assignment_response(response_text, context_matches, user_prompt):
    return {
        "response": response_text,
        "context": build_context_response(context_matches),
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


def answer_question(question):
    try:
        validate_environment()
    except Exception as error:
        raise RagPipelineError("environment validation", error) from error

    try:
        results = query_hybrid(question)
    except Exception as error:
        raise RagPipelineError("hybrid retrieval", error) from error

    try:
        context_matches = select_context_matches(question, results)
        context_string = build_context_string(context_matches)
        user_prompt = build_user_prompt(question, context_string)
    except Exception as error:
        raise RagPipelineError("context building", error) from error

    try:
        response = invoke_chat(user_prompt)
        response_text = get_response_text(response)

        if not response_text.strip():
            retry_response = invoke_chat(user_prompt)
            retry_response_text = get_response_text(retry_response)

            if retry_response_text.strip():
                response_text = retry_response_text
            else:
                response_text = (
                    "I don’t know based on the provided Medium articles data."
                )
    except Exception as error:
        if is_content_filter_error(error):
            retry_matches = context_matches[:CONTENT_FILTER_RETRY_CHUNKS]
            retry_context_string = build_context_string(retry_matches)
            retry_user_prompt = build_user_prompt(question, retry_context_string)

            try:
                retry_response = invoke_chat(retry_user_prompt)
                retry_response_text = get_response_text(retry_response)

                if not retry_response_text.strip():
                    retry_response_text = (
                        "I don’t know based on the provided Medium articles data."
                    )

                return build_assignment_response(
                    retry_response_text,
                    retry_matches,
                    retry_user_prompt,
                )
            except Exception as retry_error:
                if is_content_filter_error(retry_error):
                    return build_assignment_response(
                        "I don’t know based on the provided Medium articles data.",
                        retry_matches,
                        retry_user_prompt,
                    )

                raise RagPipelineError(
                    "chat model top-3 retry",
                    retry_error,
                    context_count=len(retry_matches),
                    context_summary=build_context_summary(retry_matches),
                ) from retry_error

        raise RagPipelineError(
            "chat model call",
            error,
            context_count=len(context_matches),
            context_summary=build_context_summary(context_matches),
        ) from error

    return build_assignment_response(response_text, context_matches, user_prompt)


def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
