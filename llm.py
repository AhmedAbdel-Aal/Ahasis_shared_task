from openai import OpenAI
from mistralai import Mistral
from typing import Dict
import dotenv
import os

dotenv.load_dotenv()


def llm_call_deepseek(prompt):
    print("backend used deepseek")
    client = OpenAI(
        api_key=os.environ["DEEP_SEEK"], base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",  # "deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content


def llm_call_openai(prompt, model="gpt-4o"):
    print(f"backend used openai - {model}")
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math, and you are evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


def llm_call_mistral(prompt, model="ministral-8b-latest"):

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    print(f"backend used mistral - {model}")
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("--> returning response")
    return chat_response.choices[0].message.content


def llm_call_llama(prompt, model="meta-llama/Meta-Llama-3.1-70B-Instruct"):
    openai = OpenAI(
        api_key=os.environ["DEEP_INFRA"],
        base_url="https://api.deepinfra.com/v1/openai",
    )
    print(f"backend used llama - {model}")
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",  # "meta-llama/Meta-Llama-3.1-8B-Instruct",#
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return chat_completion.choices[0].message.content


def llm_call_qwen(prompt, model="Qwen/QwQ-32B"):
    openai = OpenAI(
        api_key=os.environ["DEEP_INFRA"],
        base_url="https://api.deepinfra.com/v1/openai",
    )
    print(f"backend used deep_infra qwen - {model}")
    chat_completion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return chat_completion.choices[0].message.content


def router(prompt):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPEN_ROUTER"],
    )
    model = "qwen/qwq-32b"
    print(f"backend used ROUTER: qwen - {model}")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    # print('--> returning response')
    # print(completion)
    # print(completion.choices[0].message.content)

    return completion.choices[0].message.content


def llm_call_alibaba(prompt, model="Qwen/QwQ-32B"):

    client = OpenAI(
        # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
        api_key=os.getenv("ALIBABA"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    print(f"backend used ALI Baba: qwen - {model}")
    completion = client.chat.completions.create(
        model="qwq-plus",  # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("--> returning response")
    print(completion)
    return completion.choices[0].message.content


from groq import Groq


def groq(prompt):
    client = Groq(
        api_key=os.environ.get("GROQ"),
    )
    model = "qwen-qwq-32b"
    print(f"backend used GROQ: qwen - {model}")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor specialized in Math and Science evaluating tutoring interactions.",
            },
            {"role": "user", "content": prompt},
        ],
        model="qwen-qwq-32b",
    )

    return chat_completion.choices[0].message.content


def llm_call(prompt, backend="openai", model="gpt-4o-mini"):
    if backend == "openai":
        return llm_call_openai(prompt, model)
    elif backend == "deepseek":
        return llm_call_deepseek(prompt)
    elif backend == "mistral":
        return llm_call_mistral(prompt)
    elif backend == "llama":
        return llm_call_llama(prompt)
    elif backend == "qwen":
        return router(prompt)  # llm_call_alibaba(prompt)#llm_call_qwen(prompt)
    elif backend == "deep_infra":
        return llm_call_qwen(prompt)
    elif backend == "groq":
        return groq(prompt)
    else:
        raise ValueError(
            f"Invalid backend - {backend}. Supported backends are: openai, deepseek, mistral, llama."
        )
