from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionDeveloperMessageParam
from openai import OpenAI
from pydantic import BaseModel, Field
from open_ai import LM, update_due,update_usages
import os

prompt = """
This is a list of texts extracted from a html page:
```
{}
```
For each line, it's a serial of labels or attributes for a html element, sequence number represents the element.
"""

class Element(BaseModel):
    sequence_no: int = Field(description="The sequence number of the element")
    title: str = Field(description="The title of the element, it must be a complete phrase or sentence")

class View(BaseModel):
    entities: list[Element]
    topic: str=Field(description="The topic of the html page")



model_url = os.environ.get("MODEL_URL")

# It automatically looks for the OPENAI_API_KEY environment variable
client = OpenAI(
    # base_url=model_url,
    # api_key="sk-no-key-required"
    # api_key=os.environ.get("OPENAI_API_KEY")
)

if __name__ == '__main__':

    texts = """
    1.job title,work experience
    2.company,company name,work experience
    3.role description,work experience
    4.field of study,education
    5.overall result ,gpa,grade average,education
    6.field of study,education
    7.overall result ,gpa,grade average,education
    8.type to add skills,skills,skills
    """

    system_msg = ChatCompletionSystemMessageParam(role="system", content="Recognize the topic of the whole page, and generate title for each element. The title must be a complete phrase or sentence.")
    user_msg = ChatCompletionDeveloperMessageParam(role="developer", content=prompt.format(texts))

    lm = LM()
    if model_url:
        model_name = "local-model"
    else:
        model_name = lm.model_name

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[system_msg,user_msg],
        response_format=View,
    )

    if not completion or len(completion.choices) == 0:
        raise ValueError("completion error")

    # 3. Access the parsed object directly
    event = completion.choices[0].message.parsed
    print(event.topic)
    for entity in event.entities:
        print(entity.sequence_no, "." ,entity.title)

    # 4. Update usages
    if not model_url:
        print(lm.model_name)
        update_due(lm, completion.usage.total_tokens)
        update_usages()