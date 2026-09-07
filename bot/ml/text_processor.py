import math
import os
import re
from typing import List
import numpy as np
import json
import urllib.request

from bot.utils import handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(handler)

def tokenize(phrase):
    """Extract meaningful words (ignore stop words for better clustering)"""
    tokens = set(phrase.split())
    return tokens - stop_words

stop_words = {'a', 'an', 'the', 'is', 'or', 'of', 'in', 'on'}

def load_words():
    with open(os.path.dirname(os.path.abspath(__file__)) + "/words_alpha.txt") as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

def score_cluster(cluster):
    """Score based on: cluster size + total information content"""
    size_score = len(cluster) * 2  # Prefer larger clusters

    # Total unique tokens across all phrases
    all_tokens = set()
    for phrase in cluster:
        all_tokens.update(tokenize(phrase))

    token_score = len(all_tokens)

    return size_score + token_score


class Recognizer:

    def __init__(self):
        self.words = load_words()
        self.embedding_endpoint = "http://"+os.environ.get("embedding_ip", "localhost")+":8080/embed"

        # model_dir = Path(__file__).resolve().parent / "onnx_model"
        #
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_dir.as_posix(),
        #     local_files_only=True,
        # )
        # self.session = ort.InferenceSession(
        #     (model_dir / "model.onnx").as_posix(),
        #     providers=["CPUExecutionProvider"],
        # )
        # self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}

    def embeddings(self, texts: List[str]) -> np.ndarray:
        """
        curl -X POST http://0.0.0:8080/embed -H "Content-Type: application/json" -d '{"text": "Embedding testing."}'
        :param texts:
        :return:
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        token_embeddings = np.zeros(shape=(len(texts), 384),dtype=np.float32)

        for i, text in enumerate(texts):
            data = {"text": text}
            # Encode the payload to bytes
            encoded_data = json.dumps(data).encode(encoding="utf-8")

            # Create the POST request with the required headers
            req = urllib.request.Request(
                self.embedding_endpoint,
                data=encoded_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                # Send the request and process the response
                with urllib.request.urlopen(req) as response:
                    response_body = response.read().decode("utf-8")

                    # Parse the JSON response
                    response_json = json.loads(response_body)

                    # Extract the float array
                    embedding = response_json.get("embedding", [])

                    token_embeddings[i]=embedding
            except Exception as e:
                logger.error(f"HTTP Request failed: {e}")

        return token_embeddings
        # encoded = self.tokenizer(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="np",
        # )
        #
        # ort_inputs = {}
        # for input_name in self.input_names:
        #     if input_name in encoded:
        #         ort_inputs[input_name] = encoded[input_name].astype(np.int64)
        #     elif input_name == "token_type_ids":
        #         ort_inputs[input_name] = np.zeros_like(
        #             encoded["input_ids"],
        #             dtype=np.int64,
        #         )
        #
        # token_embeddings = self.session.run(None, ort_inputs)[0]
        # attention_mask = encoded["attention_mask"].astype(np.float32)[..., None]
        #
        # pooled = np.sum(token_embeddings * attention_mask, axis=1)
        # pooled /= np.maximum(np.sum(attention_mask, axis=1), 1e-9)
        #
        # norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        # return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)


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


def sort_by_len(phrases: List[str]) -> List[str]:
    objs = [Phrase(item) for item in phrases]
    objs.sort(key=lambda x: x.size, reverse=True)
    return [obj.text for obj in objs]


def cleaned_phrase(phrase: str) -> str:
    new_p = []
    for word in phrase.split(' '):
        if word in recognizer.words:
            new_p.append(word)
    return ' '.join(new_p)

def cleaned_text(phrases: List[str]) -> List[str]:
    cleaned = set()
    for phrase in sort_by_len(phrases):
        phrase = phrase.lower().strip()
        phrase = re.sub(r'\b\d+\b', '', phrase)
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        phrase = cleaned_phrase(phrase)
        if phrase and phrase not in stop_words:
            for other in cleaned:
                if phrase in other:
                    break
            else:
                cleaned.add(phrase)

    return list(cleaned)


def jaccard_similarity(phrase_a: str, phrase_b: str) -> float:
    # Normalize text to lowercase and split into sets of unique words
    set_a = set(phrase_a.lower().split())
    set_b = set(phrase_b.lower().split())

    # Handle the edge case of two empty input strings
    if not set_a and not set_b:
        return 1.0

    # Calculate overlap and total unique words
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    # Divide shared words by total unique words
    return len(intersection) / len(union)


class Phrase:
    def __init__(self, text: str, score: float=0.0):
        self.text = text
        self.score = score
        self.size = len(text.split())


def select_representative(docs: List[List[str]]) -> List[str]:
    docs = [cleaned_text(doc) for doc in docs]
    # cluster = extract_primary([phrase for phrases in docs for phrase in phrases])
    texts = [" ".join(phrases) for phrases in docs]
    result = []
    for i, phrases in enumerate(docs):

        # highest_score, best_phrase = 0, ""
        # topic_score, topic_phrase = 0, ""
        parts = []

        # token score
        def tf_idf(tok):
            # times of term appears in phrases
            appears = texts[i].count(tok)
            if len(docs) == 1:
                return appears / len(phrases)
            # numbers of documents contains token
            numbers = sum([1 for text in texts if tok in text])

            tf_score = appears / len(phrases) * math.log(len(docs) / numbers)
            return tf_score

        # def topic(tok):
        #     numbers = sum([1 for text in texts if tok in text])
        #     return numbers / len(docs)

        for phrase in phrases:
            tokens = tokenize(phrase)
            score = sum([tf_idf(token) for token in tokens])
            parts.append(Phrase(phrase, score))
            # if score > highest_score:
            #     highest_score, best_phrase = score, phrase
            # topic_sc = sum([topic(token) for token in tokens])
            # if topic_sc > topic_score:
            #     topic_score, topic_phrase = topic_sc, phrase

        # if len(docs) > 0 and len(best_phrase.split()) <= 5 and best_phrase != topic_phrase:
        #     best_phrase = f"{topic_phrase} {best_phrase}"

        if not parts:
            result.append("")
            continue

        parts.sort(key=lambda x: x.score, reverse=True)
        best_phrase = parts[0].text

        if len(parts) > 1 and len(best_phrase.split()) < 5:
            for part in parts[1:]:
                jac = jaccard_similarity(best_phrase, part.text)
                if 0.2 <= jac < 1.0:
                    best_phrase += "," + part.text
                    break

        result.append(best_phrase)

    return result









