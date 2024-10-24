import os
from pathlib import Path
from typing import Sequence,Callable
from minio import Minio
from llama_index.core import Document,SimpleDirectoryReader
from llama_index.readers.minio import MinioReader
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 自动为每个文档生成元数据字段
filename_fn = lambda filename: {"file_name": filename}

def read_from_local() -> Sequence[Document]:
    documents = []
    try:
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
        logging.info(f"Successfully read documents from {documents_dir}")
    except Exception as e:
        logging.error(f"Failed to read documents from local: {e}")

    return documents

# TODO:代码待完善、验证
def read_from_minio() -> Sequence[Document]:
    # 初始化 MinIO 客户端
    minio_client = Minio(
        "minio.example.com",  # MinIO 服务器地址
        access_key="your-access-key",
        secret_key="your-secret-key",
        secure=True  # 如果使用 HTTPS，设置为 True
    )

    # 定义存储桶名称
    bucket_name = os.getenv("MINIO_BUCKET_NAME", "my-documents-bucket")

    # 从 MinIO 获取所有文档路径并读取文档
    documents = []
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            path = obj.object_name
            reader = MinioReader(minio_client, bucket_name, path)
            content = reader.load_data()
            documents.append(Document(content))
        logging.info(f"Successfully read documents from MinIO bucket: {bucket_name}")
    except Exception as e:
        logging.error(f"Failed to read documents from MinIO: {e}")

    return documents  

def _reader_execute(reader_func: Callable[[], Sequence[Document]], source: str)-> Sequence[Document]:
    try:
        documents = reader_func()
        logging.info(f"Successfully read documents from {source}.")
        return documents
    except Exception as e:
        logging.error(f"Failed to read documents from {source}: {e}")
        raise

# 使用字典映射 source 和对应的读取函数
_source_reader_map: dict[str, Callable[[], Sequence[Document]]] = {
    "local": read_from_local,
    "minio": read_from_minio,
}

def read_docs(source: str = "local")-> Sequence[Document]:
    if source not in _source_reader_map:
        raise ValueError(f"unsupported document source: {source}")
    return _reader_execute(_source_reader_map[source], source)