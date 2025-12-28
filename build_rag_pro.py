import chromadb
from chromadb.utils import embedding_functions
import os

# ================= 配置 =================
DATA_FOLDER = "./knowledge_data"
# ⚠️ 关键修改：把切片大小改大！
# 原来是 300，现在改成 1200。
# 这样足以覆盖我们写的任何一篇 txt 文档，保证“整篇存入，整篇取出”。
CHUNK_SIZE = 1200 
# ========================================

# 1. 初始化
chroma_client = chromadb.PersistentClient(path="./rag_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 重建集合
try:
    chroma_client.delete_collection(name="garbage_knowledge")
    print("🗑️ 已清空旧知识库...")
except:
    pass
collection = chroma_client.create_collection(name="garbage_knowledge", embedding_function=emb_fn)

# 2. 读取文件
documents = []
metadatas = []
ids = []
id_counter = 0

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    exit()

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
print(f"📂 发现 {len(files)} 个文件，开始处理...")

for filename in files:
    filepath = os.path.join(DATA_FOLDER, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 因为 CHUNK_SIZE 很大，这里实际上就是把整个文件当做一个 chunk
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE]
        if len(chunk) < 10: continue 
        
        documents.append(chunk)
        metadatas.append({"source": filename})
        ids.append(f"doc_{id_counter}")
        id_counter += 1

# 3. 存入数据库
if documents:
    print(f"🧠 正在存入 {len(documents)} 条完整知识...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    print(f"✅ 成功！知识库已更新。现在每条知识都是完整的了。")