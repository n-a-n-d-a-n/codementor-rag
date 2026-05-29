"""Weak spot tracker for detecting challenging DSA topics."""
import numpy as np
from typing import Optional
from sklearn.cluster import KMeans
from sqlalchemy.orm import Session as DBSession
from backend.db import crud
from backend.config import N_CLUSTERS, MIN_SAMPLES_FOR_CLUSTERING


class WeakSpotTracker:
    """Tracks weak areas by clustering query embeddings."""
    
    def __init__(self, db: DBSession):
        """Initialize weak spot tracker with database connection."""
        self.db = db
        self.topic_keywords = {
            "array": ["array", "list", "index", "element"],
            "hashmap": ["hash", "map", "dictionary", "key", "value"],
            "two_pointers": ["pointer", "two pointer", "fast", "slow"],
            "sliding_window": ["window", "sliding", "substring", "subarray"],
            "binary_search": ["binary", "search", "sorted", "log"],
            "stack": ["stack", "lifo", "pop", "push"],
            "queue": ["queue", "fifo", "deque"],
            "linked_list": ["linked", "node", "next", "pointer"],
            "tree": ["tree", "dfs", "bfs", "traversal", "node"],
            "dynamic_programming": ["dp", "dynamic", "memoization", "tabulation"],
            "backtracking": ["backtrack", "recursion", "permutation", "combination"],
            "heap": ["heap", "priority", "min", "max"],
            "graph": ["graph", "node", "edge", "cycle", "path"],
        }
    
    def cluster_embeddings(self, embeddings: list) -> Optional[dict]:
        """Cluster query embeddings to find weak areas."""
        if not embeddings or len(embeddings) < MIN_SAMPLES_FOR_CLUSTERING:
            return None
        
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Determine optimal number of clusters
            n_clusters = min(N_CLUSTERS, len(embeddings) // 2)
            if n_clusters < 2:
                return None
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            
            # Get cluster centers and distances
            clusters = {}
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                cluster_points = embeddings_array[cluster_mask]
                center = kmeans.cluster_centers_[cluster_id]
                
                # Calculate average distance from center (lower = stronger area)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                avg_distance = float(np.mean(distances))
                
                # Normalize to 0-1 scale (1.0 = strong, 0.0 = weak)
                strength_score = 1.0 / (1.0 + avg_distance)
                
                clusters[cluster_id] = {
                    "cluster_id": int(cluster_id),
                    "size": int(np.sum(cluster_mask)),
                    "strength_score": strength_score,
                    "member_indices": np.where(cluster_mask)[0].tolist(),
                }
            
            return clusters
        except Exception as e:
            print(f"Clustering error: {str(e)}")
            return None
    
    def infer_topic(self, queries: list) -> str:
        """Infer the primary topic from a cluster of queries."""
        if not queries:
            return "general"
        
        # Combine all query texts
        combined_text = " ".join([q.get("query_text", "").lower() for q in queries])
        
        # Score each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(combined_text.count(kw) for kw in keywords)
            topic_scores[topic] = score
        
        # Return highest scoring topic
        if max(topic_scores.values()) > 0:
            return max(topic_scores, key=topic_scores.get)
        return "general"
    
    def analyze_weak_spots(self, session_id: str, queries: list, embeddings: list) -> list:
        """Analyze and store weak spots for a session."""
        clusters = self.cluster_embeddings(embeddings)
        if not clusters:
            return []
        
        weak_spots = []
        for cluster_id, cluster_info in clusters.items():
            # Get queries in this cluster
            member_indices = cluster_info["member_indices"]
            cluster_queries = [queries[i] for i in member_indices if i < len(queries)]
            
            # Infer topic
            topic = self.infer_topic(cluster_queries)
            
            # Create cluster record
            strength = cluster_info["strength_score"]
            crud.create_cluster(self.db, session_id, cluster_id, topic, strength)
            
            weak_spots.append({
                "cluster_id": cluster_id,
                "topic": topic,
                "strength_score": strength,
                "query_count": cluster_info["size"],
            })
        
        # Sort by weakness (lower strength = more weak)
        weak_spots.sort(key=lambda x: x["strength_score"])
        return weak_spots
    
    def get_weak_spots(self, session_id: str) -> list:
        """Retrieve weak spots for a session."""
        clusters = crud.get_clusters_by_session(self.db, session_id)
        return [
            {
                "topic": c.topic,
                "strength_score": c.strength_score,
                "query_count": c.query_count,
            }
            for c in clusters
        ]
