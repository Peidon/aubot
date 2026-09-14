import os
from typing import List, Dict
from bot.ml.text_processor import recognizer, select_representative, generate_titles

from bot.utils import handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(handler)

key = os.environ.get("OPENAI_API_KEY")

def source_titles(fields) -> List[str]:
    """

    :param fields: Dictionary Mapping field id -> labels (List of text phrases)
    :return: List of Text, indicate meaning of each field
    """
    docs = [field.get("labels") for field in fields]
    if key:
        return generate_titles(docs)
    return select_representative(docs)


def build_mapper(fields, source) -> Dict[str, str]:
    mapper = dict()
    for field, represent in zip(fields, source):
        # logger.info(f' source texts: {",".join(field.get("labels"))}')
        mapper[field.get("id")] = represent
    return mapper

def build_links(fields, target):
    """
    build connections from fields to titles
    :param fields: list(object)
    :param target: list(str)
    :return: dict(id, title)
    """
    source = source_titles(fields)
    logger.info("source titles:\n {}".format("\n".join(source)))
    if not isinstance(source, list) or len(source) == 0:
        return None
    if not isinstance(target, list):
        return None

    if len(target) == 0:
        return build_mapper(fields, source)

    scores = recognizer.similarities(source, target)
    for i, score in enumerate(scores):
        title, value = score
        logger.info(f'{title} -> {value}')
        if value > 0.95:
            source[i] = title

    return build_mapper(fields, source)