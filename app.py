from fastapi import FastAPI, UploadFile, File
from vector_initilizer import initilize_index, reload_index
from llm_initilizer import initilize_llm

app = FastAPI()

# 初始化LLM
initilize_llm()

# 初始化向量索引
query_engine = initilize_index()

@app.get("/")
async def read_root():
    return {"message":"Welcome to the Vivink!"}

@app.get("/ask")
async def ask(prompt: str):
    # query_engine = index.as_query_engine()
    response =  query_engine.query(prompt)
    return {"response": str(response)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 读取上传的文件内容
    file_content = await file.read()
    global query_engine
    query_engine = reload_index(file.filename,file_content)
    return {"fileName": file.filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)