from typing import Dict
from pydantic import BaseModel, Field
from google import genai
# from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionDeveloperMessageParam
# from openai import OpenAI

# from bot.agent.open_ai import LM,update_usages
from bot.utils import handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(handler)

prompt = """
This is a list of texts extracted from a html page:
```
{}
```
For each line, sequence number represents a input box, and implies its order on the html page,
the following texts are positioned around the input box.
Input boxes have the same topic are positioned in the same section.
Generate a title for each input field, the title allows each input field to be identified individually.
"""

class Element(BaseModel):
    sequence_no: int = Field(description="The sequence number of the input box")
    topic: str = Field(description="The topic of the input box on the web page")
    title: str = Field(description="The title of the input box, it's a name or question,doesn't contain punctuation or special characters.")

class View(BaseModel):
    entities: list[Element]
    topics: str=Field(description="The topics web page includes, split by comma")


# export GEMINI_API_KEY="YOUR_API_KEY"
client = genai.Client()

# It automatically looks for the OPENAI_API_KEY environment variable
# client = OpenAI()


def tailor(docs:str) -> list[Element]:
    # lm = LM()
    # system_msg = ChatCompletionSystemMessageParam(role="system",
    #                                               content="Tidy up information from web page")
    # dev_message = ChatCompletionDeveloperMessageParam(role="developer", content=prompt.format(docs))

    # completion = client.beta.chat.completions.parse(
    #     model=lm.model_name,
    #     messages=[system_msg, dev_message],
    #     response_format=View,
    #     temperature=0.0,
    #     seed=42
    # )
    #
    # if not completion or len(completion.choices) == 0:
    #     raise ValueError("completion error")
    # update_usages(lm, completion.usage.total_tokens)
    #
    # parsed = completion.choices[0].message.parsed

    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt.format(docs),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": View.model_json_schema()
        },
    )

    parsed = View.model_validate_json(interaction.output_text)
    return parsed.entities
