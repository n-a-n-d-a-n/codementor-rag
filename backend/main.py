"""FastAPI main application for CodeMentor RAG backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.db.database import init_db
from backend.routes import chat, evaluate, insights

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="CodeMentor RAG API",
    description="AI-powered DSA Interview Coach with RAG and Socratic Method",
    version="1.0.0",
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(evaluate.router)
app.include_router(insights.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CodeMentor RAG API",
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CodeMentor RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "POST /api/chat",
            "evaluate": "POST /api/evaluate",
            "insights": "GET /api/insights/{session_id}",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
