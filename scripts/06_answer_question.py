import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
TOP_K = 7
QUESTION = (
    "Find an article about writing better headlines. "
    "Provide the title and explain why it is relevant."
)

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""


def require_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


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
    )


def search_pinecone(question, top_k):
    embeddings_client = build_embeddings_client()
    query_vector = embeddings_client.embed_query(question)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    return index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )


def build_context(matches):
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


def print_context_summary(matches):
    print("RETRIEVED CONTEXT SUMMARY:")
    for rank, match in enumerate(matches, start=1):
        metadata = match.metadata or {}
        print(
            f"{rank}. score={match.score} "
            f"article_id={metadata.get('article_id')} "
            f"title={metadata.get('title')} "
            f"chunk_index={metadata.get('chunk_index')}"
        )


def main():
    load_dotenv()

    require_env_var("OPENAI_API_KEY")
    require_env_var("OPENAI_BASE_URL")
    require_env_var("PINECONE_API_KEY")
    require_env_var("PINECONE_INDEX_NAME")

    results = search_pinecone(QUESTION, TOP_K)
    context = build_context(results.matches)
    user_prompt = build_user_prompt(QUESTION, context)

    chat_client = build_chat_client()
    response = chat_client.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    )

    print("QUESTION")
    print(QUESTION)
    print("\nFINAL RESPONSE")
    print(response.content)
    print()
    print_context_summary(results.matches)
    print("\nAUGMENTED SYSTEM PROMPT")
    print(SYSTEM_PROMPT)
    print("\nAUGMENTED USER PROMPT")
    print(user_prompt)


if __name__ == "__main__":
    main()
