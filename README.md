### 简单的RAG示例
- FastAPI
- LlamaIndex
  - Vector Store(Index)
  - Embedding Model
  - LLM（Huggingface/Ollama/Customize LLM）

> 其中LLM使用的是自定义LLM，对接大模型厂商API(兼容OpenAI API)

### 特性
- 集成FastAPI，提供Restful API OK
- 支持文档上传，动态加载和更新向量索引 OK
- 根据配置或参数来指定Embedding Model OK
- 支持Huggingface/Ollama/Customize LLM OK
- 根据配置或参数选择Huggingface/Ollama/Customize LLM
- 根据配置或参数，灵活支持多种向量数据库
- 应用启动后从文档存储服务中读取原始文档数据，支持：MinIO, MongDB, 云存储( AWS S3，阿里云...)
- 缓存机制，减少对大模型的请求频率
- 其它特性待定

### 示例运行说明
- 1. 安装各种Python库(参考requirements.txt，或使用工具生成requirements.txt)
- 2. 在项目根目录创建"data"目录，放入你想使用的文本文件；(支持的其它格式的文件，请查看LlamaIndex官方文档，并在代码中使用对应格式的数据连接器(XXReader))
- 3. 设置环境变量，见".env"文件中的注释
- 4. 执行应用启动命令：python3 app.py，见".env"文件中的注释
- 5. 提问: http://127.0.0.1:8000/ask?prompt={实际的prompt}
- 6. 上传文件更新知识库(.txt): http://127.0.0.1:8000/upload
- 7. 执行第5.步，针对6.中上传的文件内容进行提问、验证

### 待解决的重要问题
- 提问VectorStoreIndex中不存在的内容时，有不使用LLM回答的情况出现
- 响应时间较长，尤其是上传新文件更新VectorStoreIndex之后