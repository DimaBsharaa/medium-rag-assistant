from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import RagPipelineError, answer_question, stats

app = FastAPI(title="Medium Article RAG Assistant")


@app.get("/")
def root():
    return {
        "message": "Medium RAG Assistant is running.",
        "endpoints": {
            "stats": "/api/stats",
            "prompt": "/api/prompt",
        },
    }


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return answer_question(question)
    except RagPipelineError as error:
        detail = {
            "message": str(error),
            "step": error.step,
            "error_type": type(error.original_error).__name__,
            "error": str(error.original_error),
            "context_count": error.context_count,
            "context_summary": error.context_summary,
        }
        raise HTTPException(status_code=500, detail=detail) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/stats")
def get_stats():
    return stats()
