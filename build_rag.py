import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化数据库 (保存在本地文件夹 ./rag_db)
chroma_client = chromadb.PersistentClient(path="./rag_db")

# 2. 使用开源的轻量级 Embedding 模型 (会自动下载，很小)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. 创建/重置集合
try:
    chroma_client.delete_collection(name="garbage_knowledge")
except:
    pass
collection = chroma_client.create_collection(name="garbage_knowledge", embedding_function=emb_fn)

# 4. 准备你的“海量”知识 (这里是你的创新点，可以去网上找环保法规往里塞)
documents = [
    # 电池类知识
    "废旧电池含有汞、铅、镉等重金属，如果随意丢弃会通过土壤渗透污染地下水。",
    "纽扣电池、充电电池属于有害垃圾，必须投放到红色的有害垃圾收集容器中。",
    "普通干电池（如碱性电池）现在已实现低汞或无汞化，在部分城市可作为其他垃圾处理，但建议仍按有害垃圾投放以防万一。",
    
    # 塑料类知识
    "塑料瓶属于可回收物，投放前建议倒空瓶内液体，并压扁以减少体积。",
    "被污染的塑料饭盒（如沾满油渍）很难回收，通常应归类为其他垃圾（干垃圾）。",
    "塑料降解需要数百年时间，焚烧会产生二恶英等致癌气体，回收利用是最佳方案。",
    
    # 金属类知识
    "易拉罐是回收价值很高的金属，可以无限次熔炼再生。",
    "尖锐的金属废弃物（如刀片、钉子）在投放前应用纸包裹，避免伤害环卫工人。",
    
    # 通用环保知识
    "垃圾分类遵循'能卖拿去卖，有害单独放，干湿要分开'的原则。",
    "不同城市的垃圾分类标准可能略有不同，如上海分干湿，北京分厨余和其他。",
]

# 5. 写入数据库
ids = [f"id_{i}" for i in range(len(documents))]
metadatas = [{"source": "环保手册"} for _ in range(len(documents))]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ 成功构建 RAG 知识库！共存入 {len(documents)} 条知识。")
print("💾 数据库路径: ./rag_db")