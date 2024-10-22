import os
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from customized_llm import CustomizedLLM
from const import query_wrapper_prompt

def huggingface_llm(model_name=os.getenv("HUGGINGFACE_MODEL")) -> HuggingFaceLLM:
    print(f"The local model currently in use is {model_name}")
    return  HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        query_wrapper_prompt=query_wrapper_prompt,
        # generate_kwargs={"do_sample": True},
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        device_map="auto",
    )

def ollama_llm(model_name=os.getenv("OLLAMA_MODEL")):
    # 1. 检测本地ollama服务是否启动？是否运行模型？ todo
    # 2. 若没有，则执行脚本，启动ollama服务，并运行指定LLM todo
    print(f"The ollama model currently in use is {model_name}")
    return Ollama(model=model_name,request_timeout=360.0)


def initilize_llm():
    # 指定LLM todo: 根据配置使用不同的方式，不同的LLM
    Settings.llm = CustomizedLLM() # huggingface_llm() OR ollama_llm()
    # 指定Embedding Model
    Settings.embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))