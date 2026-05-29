"""Retriever module for RAG-based coaching."""
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from backend.config import GOOGLE_API_KEY, LLM_MODEL, CONVERSATION_WINDOW_SIZE
from backend.pipeline.ingest import load_faiss_index


COACH_PROMPT_TEMPLATE = """You are an expert DSA coach committed to the Socratic method. Your role is to guide students toward solutions, never giving away answers directly.

IMPORTANT RULES:
1. NEVER provide full solutions or complete code
2. ALWAYS ask what the student has tried
3. ALWAYS encourage the student to think about:
   - Time complexity implications
   - Space complexity trade-offs
   - Edge cases and boundary conditions
4. Ask clarifying questions like:
   - "What approach have you considered?"
   - "What's the time complexity of your current approach?"
   - "What edge cases might break this?"
5. If the student seems stuck after hints, guide them incrementally
6. Always mention time and space complexity considerations

Context from similar problems:
{context}

Student question: {question}

Coach response (Socratic, guiding, never revealing full answers):"""


class SocraticCoach:
    """RAG-based Socratic coach for DSA."""
    
    def __init__(self):
        """Initialize the coach with FAISS retriever and LLM."""
        # Load FAISS index
        self.vector_store = load_faiss_index()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 similar problems
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=1024,
        )
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=CONVERSATION_WINDOW_SIZE,
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=COACH_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            prompt=self.prompt,
            return_source_documents=True,
            verbose=False,
        )
    
    def coach(self, question: str) -> dict:
        """Process a student question and return coaching response."""
        try:
            # Add to memory
            self.memory.save_context(
                {"input": question},
                {"output": ""}
            )
            
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract source documents
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    sources.append({
                        "title": doc.metadata.get("title", "Unknown"),
                        "difficulty": doc.metadata.get("difficulty", "Unknown"),
                        "pattern": doc.metadata.get("pattern", "Unknown"),
                    })
            
            response = result.get("result", "I encountered an error processing your question.")
            
            # Update memory with actual response
            self.memory.save_context(
                {"input": question},
                {"output": response}
            )
            
            return {
                "response": response,
                "sources": sources,
                "success": True,
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "success": False,
            }
    
    def get_conversation_history(self) -> list:
        """Get current conversation history from memory."""
        if hasattr(self.memory, 'buffer'):
            return self.memory.buffer
        return []
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
