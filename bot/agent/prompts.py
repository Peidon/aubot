from bs4 import BeautifulSoup
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionDeveloperMessageParam

prompt = """
This is the html text:
```
{}
```
There are sort of elements like Input box, check box, dropdown menu,
which contains information that users input.
"""

# List of CSS-related attributes to strip
css_attributes = ['class', 'style']

from openai import OpenAI
from pydantic import BaseModel, Field


class Element(BaseModel):
    key: str = Field(description="The meaning of input, usually from one label or attributes. It should be complete, and contain context. If it's hard to describe it by only one label, you need to refer to elements around it")
    value: str = Field(description="The content of input")

class Knowledge(BaseModel):
    pair: list[Element]
    topic: str = Field(description="The topic of this html content")


# It automatically looks for the OPENAI_API_KEY environment variable
client = OpenAI(
    base_url="http://192.168.0.7:8080/v1",
    api_key="sk-no-key-required"
)

if __name__ == '__main__':
    with open("/Users/pedro/program/kit/internal/form-bot/autofill-extension/demo_experience.html", "r",
              encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Remove specific tags (svg, picture, canvas, video)
    tags_to_remove = ['svg', 'picture', 'canvas', 'video']
    for tag in soup(tags_to_remove):
        tag.decompose()  # Completely removes the tag and its contents

    # 2. Remove CSS-related attributes from all remaining tags
    # Loop through all tags and delete the targeted attributes
    for tag in soup.find_all(True):
        for attr in css_attributes:
            if tag.has_attr(attr):
                del tag[attr]

    cleaned = soup.prettify()

    system_msg = ChatCompletionSystemMessageParam(role="system", content="Extract information from html")
    user_msg = ChatCompletionDeveloperMessageParam(role="developer", content=prompt.format(cleaned))

    completion = client.beta.chat.completions.parse(
        model="local-model",
        messages=[system_msg,user_msg],
        response_format=Knowledge,
    )

    # 3. Access the parsed object directly
    event = completion.choices[0].message.parsed
    print(event.pair)