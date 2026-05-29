"""Session store for managing user sessions and query history."""
from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session as DBSession
from backend.db import crud


class SessionStore:
    """Manages user sessions and query history in SQLite."""
    
    def __init__(self, db: DBSession):
        """Initialize session store with database connection."""
        self.db = db
    
    def create_session(self, session_id: str, user_name: Optional[str] = None) -> dict:
        """Create a new session."""
        session = crud.create_session(self.db, session_id, user_name)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "user_name": session.user_name,
        }
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session by ID."""
        session = crud.get_session(self.db, session_id)
        if not session:
            return None
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "user_name": session.user_name,
        }
    
    def add_query(self, session_id: str, query_text: str, embedding: list, response: Optional[str] = None) -> dict:
        """Add a query to session history."""
        query = crud.create_query(self.db, session_id, query_text, embedding, response)
        return {
            "id": query.id,
            "session_id": query.session_id,
            "query_text": query.query_text,
            "response": query.response,
            "created_at": query.created_at.isoformat(),
        }
    
    def get_session_queries(self, session_id: str) -> list:
        """Get all queries for a session."""
        queries = crud.get_queries_by_session(self.db, session_id)
        return [
            {
                "id": q.id,
                "session_id": q.session_id,
                "query_text": q.query_text,
                "response": q.response,
                "embedding": q.embedding,
                "created_at": q.created_at.isoformat(),
            }
            for q in queries
        ]
    
    def get_session_embeddings(self, session_id: str) -> list:
        """Get all embeddings for a session as a 2D list."""
        queries = crud.get_queries_by_session(self.db, session_id)
        embeddings = []
        for q in queries:
            if q.embedding:
                embeddings.append(q.embedding)
        return embeddings
