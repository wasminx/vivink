import os
import logging
from openai import OpenAI
from typing import Any, Generator
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from functools import cached_property
from const import SYSTEM_PROMPT

#配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#从环境变量获取API Key
load_dotenv()
API_KEY = os.getenv("CUSTOMIZED_LLM_APIKEY")
BASE_URL = os.getenv("CUSTOMIZED_LLM_URL")
if not API_KEY:
    raise ValueError("environment variable is not set:DEEPSEEK_APIKEY")

class CustomizedChat(BaseModel):

    api_key: str = Field(default=API_KEY)
    base_url: str = Field(default=BASE_URL)
    system_message: str
    model: str
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)

    class Config:
        """Pydantic 配置类，允许Chat模型接受任意类型的字段"""
        arbitrary_types_allowed = True

    @cached_property
    def client(self) -> OpenAI:
        # 以兼容OpenAI API的方式访问
        return OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def chat(self, 
             user_message: str,
             stream: bool = False,
             ) -> Any:
        print(f"The customized chat uses {self.model} as the model.")
        try:
            response = self.client.chat.completions.create(
                model= self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=  self.max_tokens,
                temperature= self.temperature,
                stream= stream,
            )
            return self._stream_response(response) if stream else response.choices[0].message.content
        except Exception as e:
            logger.error(f"API(${self.base_url}) call failed:\n{e}")
            raise

    def _stream_response(self, response) -> Generator[str, None, None]:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class CustomizedLLM(CustomLLM):

    # customized_chat: CustomizedChat = Field(default_factory=CustomizedChat)
    customized_chat: CustomizedChat = Field(default_factory=lambda: CustomizedChat(
        system_message = SYSTEM_PROMPT, 
        model = os.getenv("CUSTOMIZED_LLM"),
        max_tokens = 1024, # 可配置化
        temperature = 0.7, # 可配置化
    ))

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.customized_chat.chat(user_message=prompt, stream= False)
        return CompletionResponse(text=response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.customized_chat.chat(user_message=prompt, stream= True)
        def build_response():
            response_content = ""
            for chunk in self.customized_chat._stream_response(response):
                if chunk:
                    response_content += chunk
                    yield CompletionResponse(text= response_content, delta=chunk)
        return build_response()

os.environ["TOKENIZERS_PARALLELISM"] = "false"