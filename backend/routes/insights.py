"""Insights route for weak spot analysis."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.memory.session_store import SessionStore
from backend.memory.weak_spot_tracker import WeakSpotTracker

router = APIRouter(prefix="/api", tags=["insights"])


class InsightsResponse(BaseModel):
    """Response model for insights endpoint."""
    session_id: str
    weak_spots: list
    total_queries: int
    topics_covered: list


@router.get("/insights/{session_id}")
async def get_insights(session_id: str, db: Session = Depends(get_db)) -> InsightsResponse:
    """Get insights about weak areas for a session."""
    try:
        # Get session
        store = SessionStore(db)
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get queries and embeddings
        queries = store.get_session_queries(session_id)
        embeddings = store.get_session_embeddings(session_id)
        
        if not embeddings:
            return InsightsResponse(
                session_id=session_id,
                weak_spots=[],
                total_queries=0,
                topics_covered=[],
            )
        
        # Analyze weak spots
        tracker = WeakSpotTracker(db)
        weak_spots = tracker.analyze_weak_spots(session_id, queries, embeddings)
        
        # Extract topics from weak spots
        topics = list(set([ws["topic"] for ws in weak_spots]))
        
        return InsightsResponse(
            session_id=session_id,
            weak_spots=weak_spots,
            total_queries=len(queries),
            topics_covered=topics,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights error: {str(e)}")
