"""Code evaluator for Big-O complexity analysis."""
import ast
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY, LLM_MODEL


EVALUATION_PROMPT = """You are an expert algorithm complexity analyzer. Analyze the following Python code and return a VALID JSON object (no markdown, no extra text) with these exact keys:

{
  "is_correct": boolean,
  "bugs": [list of potential bugs or issues],
  "time_complexity": "string representation like O(n log n)",
  "space_complexity": "string representation like O(n)",
  "edge_cases_missed": [list of edge cases not handled],
  "optimization": "string with optimization suggestions"
}

Code to analyze:
```python
{code}
```

Return ONLY the JSON object, no other text:"""


class CodeEvaluator:
    """Evaluates DSA code for correctness and complexity."""
    
    def __init__(self):
        """Initialize the evaluator with Gemini LLM."""
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,  # Lower temperature for analytical tasks
            max_output_tokens=512,
        )
    
    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
            return True, "Syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except Exception as e:
            return False, f"Parse Error: {str(e)}"
    
    def evaluate(self, code: str) -> dict:
        """Evaluate code for correctness and complexity."""
        # First validate syntax
        is_valid, syntax_msg = self.validate_syntax(code)
        if not is_valid:
            return {
                "is_valid": False,
                "error": syntax_msg,
                "analysis": None,
            }
        
        try:
            # Call Gemini for analysis
            prompt = EVALUATION_PROMPT.format(code=code)
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            response_text = response.content.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.rstrip("`")
                
                analysis = json.loads(response_text)
                
                # Validate required keys
                required_keys = {
                    "is_correct",
                    "bugs",
                    "time_complexity",
                    "space_complexity",
                    "edge_cases_missed",
                    "optimization"
                }
                
                if not all(key in analysis for key in required_keys):
                    return {
                        "is_valid": True,
                        "error": "Invalid response format from LLM",
                        "analysis": None,
                    }
                
                return {
                    "is_valid": True,
                    "error": None,
                    "analysis": analysis,
                }
            except json.JSONDecodeError as e:
                return {
                    "is_valid": True,
                    "error": f"Failed to parse LLM response as JSON: {str(e)}",
                    "analysis": None,
                }
        except Exception as e:
            return {
                "is_valid": True,
                "error": f"Evaluation error: {str(e)}",
                "analysis": None,
            }
