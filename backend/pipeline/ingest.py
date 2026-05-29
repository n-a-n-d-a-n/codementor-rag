"""Ingestion module for loading and embedding DSA problems."""
import json
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    SEEDS_DIR,
    FAISS_INDEX_PATH,
    ID_MAP_PATH,
)
from tqdm import tqdm


def load_seed_problems():
    """Load seed DSA problems from JSON file."""
    seed_file = SEEDS_DIR / "sample_problems.json"
    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")
    
    with open(seed_file, 'r') as f:
        problems = json.load(f)
    
    return problems


def format_problem_for_embedding(problem: dict) -> str:
    """Format a problem into a text string for embedding."""
    parts = [
        f"Title: {problem['title']}",
        f"Difficulty: {problem['difficulty']}",
        f"Tags: {', '.join(problem['tags'])}",
        f"Pattern: {problem['pattern']}",
        f"Problem: {problem['problem_statement']}",
        f"Time Complexity: {problem['time_complexity']}",
        f"Space Complexity: {problem['space_complexity']}",
    ]
    return "\n".join(parts)


def build_faiss_index():
    """Build FAISS index from seed problems and save to disk."""
    print("Loading seed problems...")
    problems = load_seed_problems()
    
    # Format problems for embedding
    texts = []
    metadatas = []
    
    print(f"Formatting {len(problems)} problems...")
    for problem in tqdm(problems, desc="Formatting"):
        text = format_problem_for_embedding(problem)
        texts.append(text)
        metadatas.append({
            "id": problem["id"],
            "title": problem["title"],
            "difficulty": problem["difficulty"],
            "tags": problem["tags"],
            "pattern": problem["pattern"],
        })
    
    # Initialize embeddings
    print("Initializing embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    
    # Create FAISS index
    print("Creating FAISS index...")
    faiss_index = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    
    # Save to disk
    print(f"Saving index to {FAISS_INDEX_PATH}...")
    faiss_index.save_local(str(FAISS_INDEX_PATH.parent), "faiss_index")
    
    # Save ID mapping
    id_map = {i: meta["id"] for i, meta in enumerate(metadatas)}
    with open(ID_MAP_PATH, 'w') as f:
        json.dump(id_map, f)
    
    print(f"✓ FAISS index built successfully with {len(problems)} problems")
    return faiss_index


def load_faiss_index():
    """Load FAISS index from disk."""
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            f"Run 'python scripts/build_index.py' first."
        )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    
    faiss_index = FAISS.load_local(
        str(FAISS_INDEX_PATH.parent),
        embeddings,
        index_name="faiss_index",
        allow_dangerous_deserialization=True,
    )
    
    return faiss_index
