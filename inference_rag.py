from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image
import os
import torch
import chromadb
from chromadb.utils import embedding_functions

# ================= RAG 配置 =================
print("📚 正在加载 RAG 向量知识库...")
chroma_client = chromadb.PersistentClient(path="./rag_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="garbage_knowledge", embedding_function=emb_fn)

# ================= 模型配置 =================
BASE_MODEL_PATH = "./Qwen2-VL-4bit"
ADAPTER_PATH = "qwen2_vl_garbage_finetune_full" 
if not os.path.exists(ADAPTER_PATH):
    ADAPTER_PATH = "qwen2_vl_garbage_finetune"

print(f"🚀 正在启动 EcoMind (适配器: {ADAPTER_PATH})...")
model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
FastVisionModel.for_inference(model)
print("✅ 系统就绪！")

# ================= 交互循环 =================
while True:
    print("\n" + "="*60)
    user_input = input("📸 请拖入垃圾图片 (q 退出): ").strip()
    if user_input.lower() in ['q', 'exit']: break
    
    image_path = user_input.replace('"', '').replace("'", "").strip()
    if ":" in image_path and "\\" in image_path:
        try:
            drive, rest = os.path.splitdrive(image_path)
            clean_rest = rest.replace('\\', '/')
            image_path = f"/mnt/{drive[0].lower()}{clean_rest}"
        except: pass
            
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在")
        continue
    try:
        image = Image.open(image_path).convert("RGB")
    except: continue

    # --- 1. 视觉识别 ---
    print("🤖 1. 视觉模型正在分析...")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What kind of garbage is this?"}]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(image, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    category_result = response.split("assistant")[-1].strip()
    
    print(f"👁️ 识别标签: \033[1;33m{category_result}\033[0m")

    # --- 2. 语义检索 (增强版) ---
    print("🧠 2. 正在检索知识库 (获取 Top-2 结果)...")
    
    # ⚠️ 关键修改：n_results=2，防止第一名匹配错误
    results = collection.query(
        query_texts=[category_result], 
        n_results=2 
    )
    
    print("\n" + "-"*20 + " 🌍 EcoMind 专家分析报告 " + "-"*20)
    
    if results['documents']:
        # 遍历所有检索到的结果
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            print(f"\n📄 [相关知识 {i+1}] (来源: {source})")
            print(f"\033[0;32m{doc}\033[0m") # 绿色打印知识内容
            print("-" * 40)
    else:
        print("未找到相关知识。")
    
    print("=" * 60)