from dashscope import api_key
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from typing import List
from langchain_core.embeddings import Embeddings
from FlagEmbedding import FlagModel

# import os
# os.environ["DASHSCOPE_API_KEY"] = "sk-6a9af15ccc8e4cddb24ed8d1aac55bc9"

"""
基本设置
"""

API_KEY = "Your_api_key"
"""LLM模型 API Key"""

LOAD_PATH = "./fileStorage"
"""指定加载文档的目录"""

VECTOR_DIR = "./vector_store"
"""指定持久化向量数据库的存储路径"""

MODEL_NAME = "deepseek-r1:1.5b"
"""指定大语言模型名称"""

EMBEDDING_MODEL_PATH = "./Embedding/bge-base-zh-v1.5"
"""本地 embedding 模型路径"""

OLLAMA_EMBEDDING_NAME = "text-embedding-v3"
"""Ollama 下载的 embedding 模型名称"""

EMBEDDING_NAME = "text-embedding-v3"


COLLECTION_NAME = "documents_qa"
"""向量数据库的集合名"""


def chat_llm():
    """LLM 聊天模型"""

    # 方式一：调用本地模型，调用 langchain_ollama 库下的 ChatOllama
    # 导入包：from langchain_ollama import ChatOllama
    # llm = ChatOllama(
    #     model=MODEL_NAME,
    #     temperature=0.1,
    #     streaming=True,
    #     callbacks=[StreamingStdOutCallbackHandler()],
    # )

    # 方式二：调用 langchain_deepseek 库下的 ChatDeepSeek 工具类
    # 导入包：from langchain_deepseek import ChatDeepSeek
    # llm = ChatDeepSeek(
    #     model="deepseek-reasoner",
    #     api_key=API_KEY,
    #     base_url="https://api.deepseek.com/v1",
    # )

    # 方式三：调用 langchain_openai 库下的 ChatOpenAI 工具类
    # 导入包：from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(
    #     model="deepseek-reasoner",
    #     api_key=API_KEY,
    #     base_url="https://api.deepseek.com/v1",
    #     callbacks=[StreamingStdOutCallbackHandler()],
    # )
    
    # 使用通义的qwen系列模型
    llm = ChatTongyi(
        model="qwen-turbo",
        api_key=API_KEY,
        base_url="https://api.deepseek.com/v1",
        callbacks=[StreamingStdOutCallbackHandler()],
    )


    return llm


def chroma_vector_store():
    """Chroma 向量数据库"""

    return Chroma(
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model(),
    )


def embeddings_model():
    """Embedding 模型"""

    # 方式一：调用 Ollama 服务的 embedding 模型，使用下载量第一的 nomic-embed-text embedding 模型
    # 导入包：from langchain_ollama import OllamaEmbeddings
    # https://ollama.com/library/nomic-embed-text

    # embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_NAME)

    # 以下方式使用 bge-base-zh-v1.5 embedding 模型，请前往 HuggingFace 下载：
    # https://huggingface.co/BAAI/bge-base-zh-v1.5

    # 方式二：调用 langchain_community.embeddings 库下的 HuggingFaceBgeEmbeddings
    # 导入包： from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name=EMBEDDING_MODEL_PATH,
    # )

    # 方式三：调用 langchain_huggingface 库下的 HuggingFaceEmbeddings
    # 导入包： from langchain_huggingface import HuggingFaceEmbeddings
    # 适用于支持 HuggingFace Transformers 和 Sentence-Transformers 的 embedding 模型
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    # 方式四：自定义 Embedding 接口实现
    # embeddings = CustomEmbeddings()
    embeddings = DashScopeEmbeddings(dashscope_api_key=API_KEY, model=EMBEDDING_NAME)

    return embeddings


class CustomEmbeddings(Embeddings):
    """自定义 Embedding 接口实现"""

    def __init__(self):
        # 调用 FlagEmbedding 库下的 FlagModel
        # 导入包：from FlagEmbedding import FlagModel
        model = FlagModel(model_name_or_path=EMBEDDING_MODEL_PATH)

        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

        embeddings = [self.model.encode(x) for x in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
