from typing import Optional
import spacy
import re

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


def select_representative(cluster):
    """Choose the most descriptive phrase from the cluster"""

    # Heuristics (in priority order):

    # 1. Prefer complete questions (contain '?')
    questions = [p for p in cluster if '?' in p]
    if questions:
        return max(questions, key=len)  # Longest question

    # 2. Prefer phrases without separators (not field paths like "education 7 school name")
    clean_phrases = [p for p in cluster if '  ' not in p and not re.search(r'\d', p)]
    if clean_phrases:
        return max(clean_phrases, key=lambda p: len(tokenize(p)))  # Most descriptive

    # 3. Fall back to the longest phrase (most complete information)
    return max(cluster, key=len)


def extract_primary_meaning(phrases):
    # Step 1: Clean
    cleaned = []
    for phrase in phrases:
        phrase = phrase.lower().strip()
        phrase = re.sub(r'\b\d+\b', '', phrase)
        # phrase = re.sub(r'(yes|no|search|delete|add|input|section \w+)', '', phrase)
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        if phrase:
            cleaned.append(phrase)

    if not cleaned:
        return None

    # Step 2: Cluster
    clusters = cluster_phrases(cleaned)

    # Step 3: Pick best cluster
    best_cluster = max(clusters, key=score_cluster)

    # Step 4: Select title
    title = select_representative(best_cluster)

    return title


class Recognizer:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")


    def prepare(self, labels) -> Optional[list[str]]:
        offset = 0
        for i, label in enumerate(labels):
            doc = self.nlp(label)
            if 'NOUN' not in [token.pos_ for token in doc]:
                offset += 1

        if offset < len(labels):
            return labels[offset:]
        return None


