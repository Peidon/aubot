import spacy
from sentence_transformers import SentenceTransformer

sentence_zer = "sentencizer"

class FieldClassifier:

    def __init__(self):
        self.nlp = spacy.blank("en")
        if sentence_zer not in self.nlp.pipe_names:
            self.nlp.add_pipe(sentence_zer)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",backend="onnx")

    def similarities(self, source, target):
        """
        Connect source title with target title by calculating similarity
        :param source: source titles (string list)
        :param target: target titles (string, list)
        :return: list [tuple(string, float)]
        """
        source_embeddings = self.model.encode(source)
        target_embeddings = self.model.encode(target)
        sim = self.model.similarity(source_embeddings, target_embeddings)
        def select_target(scores):
            max_value = 0.0
            k = ""
            for i, score in enumerate(scores):
                v = score.item()
                if v > max_value:
                    max_value = v
                    k = target[i]
            return k, max_value

        return [(select_target(scores)) for scores in sim]


    def meaningfulScore(self, text: str) -> int:
        """
        score text meaningful based sort of rules
        :param text: string
        :return: score
        """
        text = (text or "").strip()
        if not text:
            return 0

        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_space]
        if not tokens:
            return 0

        alpha_tokens = [token for token in tokens if token.is_alpha]
        if not alpha_tokens:
            return 0

        stop_tokens = [token for token in alpha_tokens if token.is_stop]
        digit_tokens = [token for token in tokens if token.like_num]
        symbol_tokens = [
            token for token in tokens
            if not token.is_alpha and not token.is_punct and not token.like_num
        ]

        alpha_ratio = len(alpha_tokens) / len(tokens)
        unique_ratio = len({token.lower_ for token in alpha_tokens}) / len(alpha_tokens)
        stop_ratio = len(stop_tokens) / len(alpha_tokens)

        vowels = set("aeiouy")
        vowel_like_words = [
            token for token in alpha_tokens if any(char in vowels for char in token.lower_)
        ]
        no_vowel_words = [
            token for token in alpha_tokens
            if len(token.text) > 2 and not any(char in vowels for char in token.lower_)
        ]

        score = 0.0

        # Reward inputs that look like words/phrases instead of noise.
        score += min(len(alpha_tokens) * 6, 24)
        score += alpha_ratio * 20
        score += unique_ratio * 10
        score += (len(vowel_like_words) / len(alpha_tokens)) * 12

        # Natural-language glue words are a good signal for meaningful phrases.
        if 0 < stop_ratio < 0.8:
            score += 20
        elif stop_ratio > 0:
            score += 8
        elif len(alpha_tokens) >= 2 and alpha_tokens[0].is_title:
            score += 8

        sentences = [sent for sent in doc.sents if any(token.is_alpha for token in sent)]
        longest_sentence_words = 0
        if sentences:
            longest_sentence_words = max(
                sum(1 for token in sent if token.is_alpha) for sent in sentences
            )

        if longest_sentence_words >= 3:
            score += 18
        elif len(alpha_tokens) >= 2:
            score += 8

        if text[-1] in ".!?":
            score += 12
        if text[0].isupper():
            score += 4

        # Penalize inputs that look like ids, fragments of code, or gibberish.
        score -= min(len(digit_tokens) * 8, 24)
        score -= min(len(symbol_tokens) * 6, 18)
        score -= min(len(no_vowel_words) * 6, 18)

        if len(alpha_tokens) >= 3 and stop_ratio == 0 and text[-1] not in ".!?":
            score -= 12
        if len(alpha_tokens) >= 3 and unique_ratio < 0.6:
            score -= 10
        if text.isupper() and len(alpha_tokens) > 1:
            score -= 8

        return max(0, min(100, round(score)))


classifier = FieldClassifier()

def select_label(labels):
    h_score = 0
    candi = labels[0]
    for label in labels:
        score = classifier.meaningfulScore(label)
        if score > h_score:
            h_score = score
            candi = label
    return candi

def field_mean(field):
    if not isinstance(field, dict):
        return None

    labels = field.get("labels")
    if len(labels) == 0:
        return None

    mean = select_label(labels)
    return {
        "id": field.get("id"),
        "mean": mean
    }

def batch_fields_mean(fields):
    m = dict()
    for field in fields:
        fm = field_mean(field)
        key = fm.get("id")
        val = fm.get("mean")
        if fm:
           m[key] = val
    return m


def build_links(source, target):
    """
    build connections from fields to titles
    :param source: list(str)
    :param target: list(str)
    :return: list(str)
    """
    if not isinstance(source, list) or len(source) == 0:
        return None
    if not isinstance(target, list) or len(target) == 0:
        return None
    scores = classifier.similarities(source, target)

    for i, score in enumerate(scores):
        title, value = score
        if value > 0.5:
            source[i] = title
    return source


