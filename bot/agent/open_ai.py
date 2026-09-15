import json

class LLM:
    def __init__(self, model_name: str, tpm: int, rpm: int, token_usage: int, requests: int):
        self.model_name = model_name
        self.token_limit = tpm
        self.req_limit = rpm
        self.token_usage = token_usage
        self.requests = requests

def serialize_model(obj):
    if isinstance(obj, LLM):
        return {
            "model_name": obj.model_name,
            "token_limit": obj.token_limit,
            "req_limit": obj.req_limit,
            "token_usage": obj.token_usage,
            "requests": obj.requests
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def deserialize_model(dct):
    if "model_name" in dct and "token_limit" in dct and "req_limit" in dct:
        return LLM(model_name=dct["model_name"], tpm=dct["token_limit"], rpm=dct["req_limit"],
                   token_usage=dct["token_usage"],requests=dct["requests"])
    return dct

from pathlib import Path
llm_file = Path("llm.json")
if not llm_file.is_file():
    raise ValueError("No model file. ")

def LM() -> LLM:
    with open("llm.json", "r", encoding="utf-8") as file:
        llm = json.load(file, object_hook=deserialize_model)
        if llm.token_usage >= (llm.token_limit * 0.95) or llm.requests >= (llm.req_limit * 0.95):
            raise ValueError(f'{llm.model_name} is not available. ')
        return llm


def update_usages(lm: LLM, token_cost: int):
    with open("llm.json", "w") as lm_file:
        lm.token_usage += token_cost
        lm.requests += 1
        json.dump(lm, lm_file, default=serialize_model, indent=4)