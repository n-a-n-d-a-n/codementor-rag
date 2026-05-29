"""CRUD operations for database models."""
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import desc
from backend.db.models import Session, Query, Cluster


def create_session(db: DBSession, session_id: str, user_name: str = None, metadata: dict = None):
    """Create a new session."""
    db_session = Session(
        session_id=session_id,
        user_name=user_name,
        metadata=metadata or {}
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_session(db: DBSession, session_id: str):
    """Get session by ID."""
    return db.query(Session).filter(Session.session_id == session_id).first()


def create_query(db: DBSession, session_id: str, query_text: str, embedding: list, response: str = None):
    """Create a new query record."""
    db_query = Query(
        session_id=session_id,
        query_text=query_text,
        embedding=embedding,
        response=response
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query


def get_queries_by_session(db: DBSession, session_id: str):
    """Get all queries for a session."""
    return db.query(Query).filter(Query.session_id == session_id).order_by(Query.created_at).all()


def create_cluster(db: DBSession, session_id: str, cluster_id: int, topic: str, strength_score: float):
    """Create a new cluster record."""
    # Check if cluster already exists
    existing = db.query(Cluster).filter(
        Cluster.session_id == session_id,
        Cluster.cluster_id == cluster_id
    ).first()
    
    if existing:
        existing.query_count += 1
        existing.strength_score = strength_score
        db.commit()
        db.refresh(existing)
        return existing
    
    db_cluster = Cluster(
        session_id=session_id,
        cluster_id=cluster_id,
        topic=topic,
        strength_score=strength_score
    )
    db.add(db_cluster)
    db.commit()
    db.refresh(db_cluster)
    return db_cluster


def get_clusters_by_session(db: DBSession, session_id: str):
    """Get all clusters for a session."""
    return db.query(Cluster).filter(Cluster.session_id == session_id).order_by(desc(Cluster.query_count)).all()
