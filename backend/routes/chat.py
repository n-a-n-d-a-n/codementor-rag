"""Chat route for Socratic coaching."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config import GOOGLE_API_KEY, EMBEDDING_MODEL
from backend.db.database import get_db
from backend.memory.session_store import SessionStore
from backend.pipeline.retriever import SocraticCoach

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: list
    success: bool


# Global coach instance (shared across requests)
_coach_instance = None


def get_coach():
    """Get or create SocraticCoach instance."""
    global _coach_instance
    if _coach_instance is None:
        _coach_instance = SocraticCoach()
    return _coach_instance


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Process a student question and return Socratic coaching response."""
    try:
        # Validate inputs
        if not request.session_id or not request.message:
            raise HTTPException(status_code=400, detail="Missing session_id or message")
        
        # Get or create session
        store = SessionStore(db)
        session = store.get_session(request.session_id)
        if not session:
            store.create_session(request.session_id)
        
        # Get coach and process question
        coach = get_coach()
        result = coach.coach(request.message)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["response"])
        
        # Embed the query and store in session
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        embedding = embeddings_model.embed_query(request.message)
        store.add_query(request.session_id, request.message, embedding, result["response"])
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            success=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
