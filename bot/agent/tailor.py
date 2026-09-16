from typing import Dict

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionDeveloperMessageParam
from openai import OpenAI
from pydantic import BaseModel, Field
from bot.agent.open_ai import LM,update_usages
from bot.utils import handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(handler)

prompt = """
This is a list of texts extracted from a html page:
```
{}
```
For each line, it's a serial of texts around a input box,
sequence number represents the input box,and implies it's position on the html page.
"""

class Element(BaseModel):
    sequence_no: int = Field(description="The sequence number of the input box")
    topic: str = Field(description="The topic of the input box on the web page")
    title: str = Field(description="The title of the input box, it's a name or question."
                                   "It should be a phrase or sentence, which doesn't contain special characters."
                                   "It should contain complete meaning so it's able to be identified without referring to context.")

class View(BaseModel):
    entities: list[Element]
    topics: str=Field(description="The topics web page includes, split by comma")



# model_url = os.environ.get("MODEL_URL")

# It automatically looks for the OPENAI_API_KEY environment variable
client = OpenAI()


def tailor(docs:str) -> Dict[int, str]:
    system_msg = ChatCompletionSystemMessageParam(role="system",
                                                  content="Tidy up information from web page")
    dev_message = ChatCompletionDeveloperMessageParam(role="developer", content=prompt.format(docs))
    lm = LM()
    logger.info(f'call {lm.model_name}')
    completion = client.beta.chat.completions.parse(
        model=lm.model_name,
        messages=[system_msg, dev_message],
        response_format=View,
        temperature=0.0,
        seed=42
    )

    if not completion or len(completion.choices) == 0:
        raise ValueError("completion error")
    update_usages(lm, completion.usage.total_tokens)

    parsed = completion.choices[0].message.parsed
    for entity in parsed.entities:
        logger.info(entity.sequence_no, ">>", entity.topic)
    return dict([(entity.sequence_no,entity.title) for entity in parsed.entities])
