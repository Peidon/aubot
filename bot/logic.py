from typing import List, Dict
from bot.ml.text_processor import recognizer, select_representative

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler("info.log", maxBytes=512000)
logger = logging.getLogger(__name__)
logger.addHandler(handler)


def fields_source(fields) -> List[str]:
    """

    :param fields: Dictionary Mapping field id -> labels (List of text phrases)
    :return: List of Text, indicate meaning of each field
    """
    docs = [field.get("labels") for field in fields]
    return select_representative(docs)


def build_mapper(fields, source) -> Dict[str, str]:
    mapper = dict()
    for field, represent in zip(fields, source):
        logger.info(f'{field.get("labels")} -> {represent}')
        mapper[field.get("id")] = represent
    return mapper

def build_links(fields, target):
    """
    build connections from fields to titles
    :param fields: list(object)
    :param target: list(str)
    :return: dict(id, title)
    """
    source = fields_source(fields)
    if not isinstance(source, list) or len(source) == 0:
        return None
    if not isinstance(target, list):
        return None

    if len(target) == 0:
        return build_mapper(fields, source)

    scores = recognizer.similarities(source, target)
    for i, score in enumerate(scores):
        title, value = score
        alpha = ((len(title.split()) + len(source[i].split()))>>2) * 0.125 + 1.0
        if value * alpha > 0.7:
            source[i] = title

    return build_mapper(fields, source)


