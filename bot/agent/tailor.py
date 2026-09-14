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
For each line, it's a serial of labels or attributes for a html element,
sequence number represents the element,and implies the position on the html page.
"""

class Element(BaseModel):
    sequence_no: int = Field(description="The sequence number of the element")
    title: str = Field(description="The title of the text that html element displays."
                                   "It should be a phrase of concise description,"
                                   "and indicates the relation to other elements in the same topic")

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
1.company name,work experience
2.location,work experience
3.work experience,i currently work here
4.date section month,work experience,from,start date,input
5.date section year,work experience,from,input,start date
6.date section month,work experience,to,input,end date
7.date section year,work experience,input,to,end date
8.role description,work experience
9.school name,school or university,education
10.education,field of study
11.,education,overall result,grade average
12.education,date section year,from,input,first year attended
13.actual or expected,education,date section year,last year attended,to,input
14.school name,school or university,education
15.education,field of study
16.,education,overall result,grade average
17.education,date section year,from,input,first year attended
18.actual or expected,education,date section year,last year attended,to,input
19.type to add skills
20.web address,
21.linked in account,social network accounts
    """

    d = tailor(texts)
    for k,v in d.items():
        print(k, ".", v)
