"""
Semantic Text Clustering
Clusters text phrases based on semantic similarity using embeddings and correlation.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict

from bot.ml.text_processor import recognizer


def compute_correlation_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between embeddings.

    Args:
        embeddings: numpy array of shape (n_texts, embedding_dim)

    Returns:
        Correlation matrix of shape (n_texts, n_texts)
    """
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Correlation is cosine similarity for normalized vectors
    correlation_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    return correlation_matrix


def cluster_by_correlation(
        correlation_matrix: np.ndarray,
        n_clusters: int = None,
        distance_threshold: float = 0.5,
        linkage: str = 'average'
) -> np.ndarray:
    """
    Cluster texts based on correlation matrix.

    Args:
        correlation_matrix: Correlation matrix (n_texts, n_texts)
        n_clusters: Number of clusters (if None, uses distance_threshold)
        distance_threshold: Distance threshold for clustering (lower = fewer, tighter clusters)
        linkage: Linkage criterion ('average', 'complete', 'single')

    Returns:
        Cluster labels for each text
    """
    # Convert correlation to distance: distance = 1 - correlation
    distance_matrix = 1 - correlation_matrix

    # Ensure diagonal is zero (distance to self)
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage,
        distance_threshold=distance_threshold if n_clusters is None else None
    )

    labels = clustering.fit_predict(distance_matrix)
    return labels


def semantic_cluster(
        texts: List[str],
        n_clusters: int = None,
        distance_threshold: float = 0.5,
        model: str = "sentence-transformers",
        linkage: str = 'average',
        verbose: bool = True
) -> Dict[int, List[str]]:
    """
    Main function: Cluster text phrases by semantic similarity.

    Args:
        texts: List of text phrases to cluster
        n_clusters: Number of clusters (if None, auto-determines using distance_threshold)
        distance_threshold: Distance threshold for auto-clustering (0.0-1.0)
                          Lower = fewer, tighter clusters
                          0.3-0.5 works well for most cases
        model: Embedding model ("sentence-transformers" or "openai")
        linkage: Clustering linkage ('average', 'complete', 'single')
        verbose: Print clustering details

    Returns:
        Dictionary mapping cluster_id -> list of texts in that cluster
    """
    if len(texts) < 2:
        return {0: texts}

    # Step 1: Generate embeddings
    if verbose:
        print(f"Step 1: Generating embeddings for {len(texts)} texts using {model}...")
    embeddings = recognizer.embeddings(texts)

    # Step 2: Compute correlation matrix
    if verbose:
        print("Step 2: Computing correlation matrix...")
    correlation_matrix = compute_correlation_matrix(embeddings)

    # Step 3: Cluster based on correlation
    if verbose:
        print("Step 3: Clustering by correlation...")
    labels = cluster_by_correlation(
        correlation_matrix,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage
    )

    # Organize results
    cluster_results = {}
    for text, label in zip(texts, labels):
        if label not in cluster_results:
            cluster_results[label] = []
        cluster_results[label].append(text)

    # Print results
    if verbose:
        print(f"\nFound {len(cluster_results)} clusters:")
        for cluster_id, cluster_texts in cluster_results.items():
            print(f"  Cluster {cluster_id}: {cluster_texts}")

        # Show correlation matrix
        print("\nCorrelation matrix:")
        print("Texts:", texts)
        print(np.round(correlation_matrix, 2))

    return cluster_results