import random
import json

class LLM:
    def __init__(self, model_name: str, tpm: int, rpm: int, token_usage: int, requests: int, idx: int):
        self.model_name = model_name
        self.token_limit = tpm
        self.req_limit = rpm
        self.token_usage = token_usage
        self.requests = requests
        self.idx = idx

def serialize_model(obj):
    if isinstance(obj, LLM):
        return {
            "model_name": obj.model_name,
            "token_limit": obj.token_limit,
            "req_limit": obj.req_limit,
            "token_usage": obj.token_usage,
            "requests": obj.requests,
            "idx": obj.idx
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def deserialize_model(dct):
    if "model_name" in dct and "token_limit" in dct and "req_limit" in dct:
        return LLM(model_name=dct["model_name"], tpm=dct["token_limit"], rpm=dct["req_limit"],
                   token_usage=dct["token_usage"],requests=dct["requests"], idx=dct["idx"])
    return dct

from pathlib import Path
due_log = Path("due.log")

if due_log:
    with open("due.log", "r",encoding="utf-8") as file:
        due = file.read()
    due_set = set(due.split("\n"))

with open("llm_pool.json", "r") as file:
    llm_list = json.load(file, object_hook=deserialize_model)
    if not isinstance(llm_list, list):
        raise ValueError("load llm list error.")

def LM() -> LLM:
    models= [m for m in llm_list if m.model_name not in due_set]
    if not models:
        raise ValueError("No models available.")
    i = random.randint(0, len(models)-1)
    return models[i]

def update_due(lm: LLM, token_cost: int):
    llm_list[lm.idx].token_usage += token_cost
    llm_list[lm.idx].requests += 1
    if lm.token_usage >= (lm.token_limit * 0.95) or lm.requests >= (lm.req_limit * 0.95):
        due_set.add(lm.model_name)
        with open("due.log", "a") as log_file:
            log_file.write("{}\n".format(lm.model_name))

def update_usages():
    with open("llm_pool.json", "w") as lm_file:
        json.dump(llm_list, lm_file, default=serialize_model, indent=4)

# if __name__ == '__main__':
#     model = LM()
#     print(model.model_name)
#     llm_list[4].token_usage = 20000
#     llm_list[4].requests = 2
#     update_usages()