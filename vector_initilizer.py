from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import Sequence
from pathlib import Path
import chromadb
import os
from dotenv import load_dotenv

##===============初始化Vector Storage & Vector Index============##

load_dotenv()

# 自动为每个文档生成元数据字段
filename_fn = lambda filename: {"file_name": filename}

def initialize_index():
    # 加载数据 todo: 用文档数据库 或 OSS 代替本地目录
    default_documents_dir = Path.home() / ".vivink/documents"
    documents_dir = Path(os.getenv("LOCAL_DOCS_DIR", str(default_documents_dir)))

    if not documents_dir.exists():
        raise ValueError(f"Knowledge documents directory does not exist: {documents_dir}")
    if not documents_dir.is_dir():
        raise ValueError(f"Path is not a directory: {documents_dir}")
    if not any(documents_dir.iterdir()):
        raise ValueError(f"Directory is empty: {documents_dir}")
    
    documents = SimpleDirectoryReader(input_dir=str(documents_dir), 
                                      recursive=True,
                                      exclude_hidden=False,
                                      file_metadata=filename_fn
                                      ).load_data()
    return _get_query_engine(documents)

def reload_index(filename,file_content, file_category=None):
    # 将文件内容转换为Document对象
    metadata = {"filename": filename}

    if file_category is not None:
        metadata["category"] = file_category

    document = Document(text=file_content, metadata = metadata)    
    # 重新加载数据并更新索引
    # documents = [document]
    # 是更新还是覆盖？
    return _get_query_engine([document])

def _get_query_engine(documents: Sequence[Document]):
    # 初始化Vector Index
    storage_context = StorageContext.from_defaults(vector_store=_create_vector_store())
    index = VectorStoreIndex.from_documents(documents, storage_context= storage_context)
    
    # 保存索引
    index.storage_context.persist()
    
    retriever = VectorIndexRetriever(
        index = index,
        similarity_top_k=3 #检索前3个最相关的文档，实际应用中需要根据场景来评估
    )

    # 当Query的内容与文档无关时(即向量DB中没有相关内容)，则正常使用LLM进行响应。
    # Default: response_mode="compact"
    # 目前看起来这种情况下，没有使用到LLM todo...
    response_synthesizer = get_response_synthesizer()
    
    query_engine = RetrieverQueryEngine(
        retriever= retriever,
        response_synthesizer=response_synthesizer
    )
    return query_engine

def _create_vector_store():
    # in-memory or in-disk
    store_mode = os.getenv("VECTOR_STORE_MODE")

    if store_mode == "in-memory":
       db_client = chromadb.EphemeralClient()
    elif store_mode == "in-disk":
       db_client = chromadb.PersistentClient(path=str(Path.home()/".vivink/vector_store"))
    else:
        raise ValueError(f"unknown vector store mode:{store_mode}")

    data_collection = db_client.get_or_create_collection("vivink-collection")

    vector_store = ChromaVectorStore(chroma_collection=data_collection)

    return vector_store