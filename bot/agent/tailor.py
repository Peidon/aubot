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
For each line, it's a serial of labels or attributes for a html element, sequence number represents the element.
"""

class Element(BaseModel):
    sequence_no: int = Field(description="The sequence number of the element")
    title: str = Field(description="The title of the element, it should be a complete and concise phrase or sentence.")

class View(BaseModel):
    entities: list[Element]
    topic: str=Field(description="The topic of the html page")



# model_url = os.environ.get("MODEL_URL")

# It automatically looks for the OPENAI_API_KEY environment variable
client = OpenAI(
    # base_url=model_url,
    # api_key="sk-no-key-required"
    # api_key=os.environ.get("OPENAI_API_KEY")
)


def tailor(docs:str) -> Dict[int, str]:
    system_msg = ChatCompletionSystemMessageParam(role="system",
                                                  content="Recognize the topic of the whole page, and generate title for each element.")
    dev_message = ChatCompletionDeveloperMessageParam(role="developer", content=prompt.format(docs))
    lm = LM()
    logger.info(f'call {lm.model_name}')
    completion = client.beta.chat.completions.parse(
        model=lm.model_name,
        messages=[system_msg, dev_message],
        response_format=View,
    )

    if not completion or len(completion.choices) == 0:
        raise ValueError("completion error")
    update_usages(lm, completion.usage.total_tokens)

    parsed = completion.choices[0].message.parsed
    return dict([(entity.sequence_no,entity.title) for entity in parsed.entities])


if __name__ == '__main__':

    texts = """
    0.job title,work experience
    1.company,company name,work experience
    3.role description,work experience
    4.field of study,education
    5.overall result ,gpa,grade average,education
    6.field of study,education
    7.overall result ,gpa,grade average,education
    8.type to add skills,skills,skills
    """

    d = tailor(texts)
    for k,v in d.items():
        print(k, ".", v)
