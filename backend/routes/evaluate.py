"""Code evaluation route for complexity analysis."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.pipeline.evaluator import CodeEvaluator

router = APIRouter(prefix="/api", tags=["evaluate"])


class EvaluateRequest(BaseModel):
    """Request model for code evaluation."""
    session_id: str
    code: str
    language: str = "python"


class EvaluateResponse(BaseModel):
    """Response model for code evaluation."""
    is_valid: bool
    error: str = None
    analysis: dict = None


# Global evaluator instance
_evaluator_instance = None


def get_evaluator():
    """Get or create CodeEvaluator instance."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = CodeEvaluator()
    return _evaluator_instance


@router.post("/evaluate")
async def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)) -> EvaluateResponse:
    """Evaluate Python code for correctness and complexity."""
    try:
        # Only support Python for now
        if request.language.lower() != "python":
            raise HTTPException(status_code=400, detail="Only Python is supported")
        
        if not request.code or not request.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        
        # Evaluate code
        evaluator = get_evaluator()
        result = evaluator.evaluate(request.code)
        
        return EvaluateResponse(
            is_valid=result["is_valid"],
            error=result["error"],
            analysis=result["analysis"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
