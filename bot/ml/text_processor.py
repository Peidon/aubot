import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Set

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from sklearn.cluster import AgglomerativeClustering

def tokenize(phrase):
    """Extract meaningful words (ignore stop words for better clustering)"""
    tokens = set(phrase.split())
    return tokens - stop_words


def cluster_phrases(phrases):
    """Group phrases by shared tokens"""
    clusters = []
    used = set()

    for i, phrase in enumerate(phrases):
        if i in used:
            continue

        tokens_i = tokenize(phrase)
        cluster = [phrase]
        used.add(i)

        # Find similar phrases (share >50% of tokens)
        for j, other_phrase in enumerate(phrases[i + 1:], start=i + 1):
            if j in used:
                continue

            tokens_j = tokenize(other_phrase)
            overlap = len(tokens_i & tokens_j)
            min_size = min(len(tokens_i), len(tokens_j))

            if overlap > 0 and overlap / min_size >= 0.5:
                cluster.append(other_phrase)
                used.add(j)

        clusters.append(cluster)

    return clusters


stop_words = {'a', 'an', 'the', 'is', 'or', 'of', 'in', 'to'}

def score_cluster(cluster):
    """Score based on: cluster size + total information content"""
    size_score = len(cluster) * 2  # Prefer larger clusters

    # Total unique tokens across all phrases
    all_tokens = set()
    for phrase in cluster:
        all_tokens.update(tokenize(phrase))

    token_score = len(all_tokens)

    return size_score + token_score


def select_representative(docs: List[List[str]], phrases: List[str]) -> str:
    """Choose the most descriptive phrase from the cluster"""

    # Heuristics (in priority order):
    mx_score, best = 0, ""
    text = " ".join(phrases)
    for phrase in phrases:
        tokens = tokenize(phrase)
        token_score = sum([text.count(token) for token in tokens])

    return best


class Recognizer:

    def __init__(self):
        model_dir = Path(__file__).resolve().parent / "onnx_model"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir.as_posix(),
            local_files_only=True,
        )
        self.session = ort.InferenceSession(
            (model_dir / "model.onnx").as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}

    def embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        ort_inputs = {}
        for input_name in self.input_names:
            if input_name in encoded:
                ort_inputs[input_name] = encoded[input_name].astype(np.int64)
            elif input_name == "token_type_ids":
                ort_inputs[input_name] = np.zeros_like(
                    encoded["input_ids"],
                    dtype=np.int64,
                )

        token_embeddings = self.session.run(None, ort_inputs)[0]
        attention_mask = encoded["attention_mask"].astype(np.float32)[..., None]

        pooled = np.sum(token_embeddings * attention_mask, axis=1)
        pooled /= np.maximum(np.sum(attention_mask, axis=1), 1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)


    def similarities(self, source, target):
        """
        Connect source title with target title by calculating similarity
        :param source: source titles (string list)
        :param target: target titles (string, list)
        :return: list [tuple(string, float)]
        """
        if isinstance(source, str):
            source = [source]
        if isinstance(target, str):
            target = [target]
        if not source or not target:
            return []

        source_embeddings = self.embeddings(source)
        target_embeddings = self.embeddings(target)
        similarities = np.matmul(source_embeddings, target_embeddings.T)
        best_match_indexes = np.argmax(similarities, axis=1)

        return [
            (target[target_index], float(similarities[source_index, target_index]))
            for source_index, target_index in enumerate(best_match_indexes)
        ]

recognizer = Recognizer()

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
        linkage: str = 'average',
        verbose: bool = False
) -> Dict[int, List[str]]:
    """
    Main function: Cluster text phrases by semantic similarity.

    Args:
        texts: List of text phrases to cluster
        n_clusters: Number of clusters (if None, auto-determines using distance_threshold)
        distance_threshold: Distance threshold for auto-clustering (0.0-1.0)
                          Lower = fewer, tighter clusters
                          0.3-0.5 works well for most cases
        linkage: Clustering linkage ('average', 'complete', 'single')
        verbose: Print clustering details

    Returns:
        Dictionary mapping cluster_id -> list of texts in that cluster
    """
    if len(texts) < 2:
        return {0: texts}

    # Step 1: Generate embeddings
    if verbose:
        print(f"Step 1: Generating embeddings for {len(texts)} texts ...")
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


def extract_primary(phrases: List[str]) -> List[str]:
    # Step 1: Clean
    cleaned = []
    for phrase in phrases:
        phrase = phrase.lower().strip()
        phrase = re.sub(r'\b\d+\b', '', phrase)
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        if phrase:
            cleaned.append(phrase)

    if not cleaned:
        return []

    # Step 2: Cluster
    clusters = semantic_cluster(cleaned,n_clusters=2)

    # Step 3: Pick best cluster
    return max(clusters.values(), key=score_cluster)


def extract_representative(docs: List[List[str]]) -> List[str]:
    docs = [extract_primary(doc) for doc in docs]
    texts = [" ".join(phrases) for phrases in docs]
    result = []
    for i, phrases in enumerate(docs):

        highest_score, best_phrase = 0, ""

        for phrase in phrases:
            tokens = tokenize(phrase)

            # token score
            def tf_idf(to):
                # times of term appears in phrases
                appears = texts[i].count(to)
                # numbers of documents contains token
                numbers = sum([1 for text in texts if to in text])
                return appears / len(phrases) * math.log(len(docs) / numbers)

            score = sum([tf_idf(token) for token in tokens])
            if score > highest_score:
                highest_score = score
                best_phrase = phrase

        result.append(best_phrase)

    return result









