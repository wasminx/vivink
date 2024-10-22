from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import Sequence
from pathlib import Path

##===============初始化Vector Storage & Vector Index============##

"""
此处需要注意dim参数:
需要明确当前Embedding Model的输出向量维度，比如BAAI/bge-large-zh-v1.5输出的维度是1024，则向量大小是:1024*4字节=4096.
在初始化VectorStore时指定的dim，若dim值乘以4字节后的结果与使用的Embedding Model输出的向量大小要相同,
即：VectorStore的向量维度 与 Embedding Model的输出向量维度要一致，否则VectorStore初始化失败，报异常：
message=vector dimension mismatch, expected vector size(byte) XX, actual 4096.
todo: 在对接其它向量数据库时 再验证一下这个场景
todo: 如何根据指定Embedding Model动态获取起向量维度？？
todo: 用Server 或 Cloud代替Local file
"""

vector_store = MilvusVectorStore(uri=str(Path.home()/".milvus/data/milvus_demo.db"), dim=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 自动为每个文档生成元数据字段
filename_fn = lambda filename: {"file_name": filename}

def initilize_index():
    # 加载数据 todo: 用文档数据库 或 OSS 代替本地目录
    documents = SimpleDirectoryReader("./data", file_metadata=filename_fn).load_data()
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



