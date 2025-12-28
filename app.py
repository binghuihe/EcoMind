import streamlit as st
from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image
import torch
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import plotly.express as px
import time
import folium
from streamlit_folium import st_folium

# ================= 1. 全局配置与黑金 CSS =================
st.set_page_config(
    page_title="EcoMind 城市大脑",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; color: #e0e0e0; }
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #00e676 !important; }
    
    /* 垃圾桶卡片 */
    .bin-card { padding: 25px; border-radius: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 0 20px rgba(0,0,0,0.4); }
    .bin-blue { background: linear-gradient(135deg, #1565C0, #0D47A1); border: 2px solid #42A5F5; }
    .bin-red { background: linear-gradient(135deg, #C62828, #B71C1C); border: 2px solid #EF5350; }
    .bin-green { background: linear-gradient(135deg, #2E7D32, #1B5E20); border: 2px solid #66BB6A; }
    .bin-gray { background: linear-gradient(135deg, #424242, #212121); border: 2px solid #BDBDBD; }
    
    /* RAG 知识框 */
    .rag-box {
        background-color: #161b22;
        border-left: 5px solid #00e676;
        padding: 20px;
        border-radius: 5px;
        margin-top: 10px;
        color: #c9d1d9;
        font-family: monospace;
        white-space: pre-wrap; /* 保持换行格式 */
    }

    .stButton>button { background: linear-gradient(90deg, #00c853, #64dd17); color: #000; font-weight: bold; border: none; }
    </style>
""", unsafe_allow_html=True)

# ================= 2. 模型加载 =================
@st.cache_resource
def load_resources():
    print("🚀 正在加载模型...")
    # 1. 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        "./Qwen2-VL-4bit",
        load_in_4bit=True,
    )
    # 加载 LoRA
    model = PeftModel.from_pretrained(model, "qwen2_vl_garbage_finetune_full")
    FastVisionModel.for_inference(model)
    
    # 2. 加载 RAG
    client = chromadb.PersistentClient(path="./rag_db")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="garbage_knowledge", embedding_function=emb_fn)
    
    return model, tokenizer, collection

try:
    model, tokenizer, collection = load_resources()
except Exception as e:
    st.error(f"核心资源加载失败: {e}")
    st.stop()

# ================= 3. 核心功能函数 =================

def get_bin_guide(cat):
    cat = cat.lower()
    if any(x in cat for x in ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'clothes', 'shoe', 'electronic', 'book']):
        return { "style": "bin-blue", "name": "可回收物", "icon": "♻️", "action": "请投入 蓝色 垃圾桶" }
    elif any(x in cat for x in ['battery', 'hazardous', 'medical', 'medicine', 'light', 'chemical', 'paint']):
        return { "style": "bin-red", "name": "有害垃圾", "icon": "☣️", "action": "请投入 红色 垃圾桶" }
    elif any(x in cat for x in ['biological', 'food', 'fruit', 'vegetable', 'plant', 'leftover', 'meal']):
        return { "style": "bin-green", "name": "厨余/湿垃圾", "icon": "🍂", "action": "请投入 绿色 垃圾桶" }
    else:
        return { "style": "bin-gray", "name": "其他垃圾", "icon": "🗑️", "action": "请投入 黑色 垃圾桶" }

def run_inference_simple(image):
    """
    极简推理：只做识别，不聊天。
    返回：类别、RAG原文、来源
    """
    # 1. 视觉识别 Prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What kind of garbage is this? (Answer succinctly)"}
        ]}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(image, text_prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    
    # 2. 生成 (不需要太长，只出类别)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    # 3. 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 后的内容
    if "assistant" in response:
        category = response.split("assistant")[-1].strip()
    else:
        category = response.strip()

    # 清洗掉可能多余的描述，只取第一行
    category = category.split('\n')[0]
        
    # 4. RAG 查库 (直接查，不生成建议)
    results = collection.query(query_texts=[category], n_results=1)
    if results['documents'] and results['documents'][0]:
        knowledge = results['documents'][0][0]
        source = results['metadatas'][0][0]['source']
    else:
        knowledge = "暂无具体工业标准，请按一般生活垃圾处理。"
        source = "通用知识库"
        
    return category, knowledge, source

# ================= 4. UI 逻辑 =================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/recycling-symbol.png", width=90)
    st.markdown("### EcoMind")
    menu = st.radio("导航", ["📸 AI 智能识别", "📊 数据驾驶舱", "🗺️ 赛博回收地图", "💰 积分黑市"])
    st.success("🟢 System Online")
    if 'points' not in st.session_state: st.session_state.points = 1250
    st.info(f"💎 积分: {st.session_state.points}")

if menu == "📸 AI 智能识别":
    st.title("📸 智能分类终端")
    
    # 初始化
    if "curr_img" not in st.session_state: st.session_state.curr_img = None
    if "res_meta" not in st.session_state: st.session_state.res_meta = None

    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.markdown("#### 1. 采集")
        src = st.radio("来源", ["上传", "拍照"], horizontal=True, label_visibility="collapsed")
        file = st.file_uploader("img") if src == "上传" else st.camera_input("cam")
        
        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="View", use_container_width=True)
            
            # 按钮点击后，只做单纯的推理，不聊天
            if st.button("🚀 开始分析", use_container_width=True):
                with st.spinner("视觉矩阵解码中..."):
                    start_t = time.time()
                    st.session_state.curr_img = img
                    
                    # 调用极简推理
                    cat, know, src = run_inference_simple(img)
                    
                    end_t = time.time()
                    st.session_state.res_meta = {
                        "c": cat, 
                        "k": know, 
                        "s": src, 
                        "latency": end_t - start_t
                    }

    with c2:
        st.markdown("#### 2. 处置指引")
        
        if st.session_state.res_meta:
            meta = st.session_state.res_meta
            guide = get_bin_guide(meta['c'])
            
            # [A] 结果卡片 (保持酷炫)
            st.markdown(f"""
            <div class="bin-card {guide['style']}">
                <div style="font-size:3rem">{guide['icon']}</div>
                <h2>{guide['name']}</h2>
                <p style="font-size:1.2rem; font-weight:bold;">{guide['action']}</p>
                <p style="font-size:0.8rem; opacity:0.8;">识别结果: {meta['c']} | 耗时: {meta['latency']:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            # [B] 直接显示 RAG 知识 (这正是你截图里想要的效果)
            st.markdown(f"#### 📖 协议指南 (Source: {meta['s']})")
            st.markdown(f"""
            <div class="rag-box">
{meta['k']}
            </div>
            """, unsafe_allow_html=True)

            # 积分按钮
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 归档并获取积分 (+10)", use_container_width=True):
                st.session_state.points += 10
                st.toast("✅ 积分 +10", icon="🎉")
                time.sleep(1)
                st.rerun()

# --- 其他模块 (保持功能完整) ---
elif menu == "📊 数据驾驶舱":
    st.title("📊 数据驾驶舱")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="metric-card"><h3 style="color:#64dd17">4,285</h3><p>今日吞吐量</p></div>', unsafe_allow_html=True)
    m2.markdown('<div class="metric-card"><h3 style="color:#00b0ff">32.4%</h3><p>资源化率</p></div>', unsafe_allow_html=True)
    m3.markdown('<div class="metric-card"><h3 style="color:#ff1744">128</h3><p>有害拦截</p></div>', unsafe_allow_html=True)
    m4.markdown('<div class="metric-card"><h3 style="color:#ffea00">8,942</h3><p>在线节点</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("构成光谱")
        df = pd.DataFrame({'类别':['可回收','厨余','其他','有害'], '数值':[45,30,20,5]})
        st.plotly_chart(px.pie(df, values='数值', names='类别', template="plotly_dark"), use_container_width=True)
    with col2:
        st.subheader("流量监控")
        df2 = pd.DataFrame({'时间':['8:00','12:00','18:00'], '负载':[20,80,90]})
        st.plotly_chart(px.bar(df2, x='时间', y='负载', template="plotly_dark"), use_container_width=True)

elif menu == "🗺️ 赛博回收地图":
    st.title("🗺️ 回收地图")
    if 'user_pos' not in st.session_state: st.session_state.user_pos = [31.2304, 121.4737]
    m = folium.Map(location=st.session_state.user_pos, zoom_start=15, tiles='CartoDB dark_matter')
    folium.Marker([31.2314, 121.4747], popup="智能柜", icon=folium.Icon(color="green", icon="leaf")).add_to(m)
    folium.Marker(st.session_state.user_pos, popup="YOU", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    st_folium(m, height=400, width="100%")

elif menu == "💰 积分黑市":
    st.title("💰 积分兑换")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://img.icons8.com/fluency/480/000000/coffee-to-go.png", width=100)
        if st.button("咖啡 (500分)", key="b1", use_container_width=True):
            if st.session_state.points >= 500:
                st.session_state.points -= 500
                st.toast("兑换成功！")
                time.sleep(1)
                st.rerun()
            else: st.error("积分不足")
    with c2:
        st.image("https://img.icons8.com/fluency/480/000000/subway.png", width=100)
        if st.button("地铁票 (800分)", key="b2", use_container_width=True):
            if st.session_state.points >= 800:
                st.session_state.points -= 800
                st.toast("兑换成功！")
                time.sleep(1)
                st.rerun()
            else: st.error("积分不足")
    with c3:
        st.image("https://img.icons8.com/fluency/480/000000/soap.png", width=100)
        if st.button("洗衣液 (300分)", key="b3", use_container_width=True):
            if st.session_state.points >= 300:
                st.session_state.points -= 300
                st.toast("兑换成功！")
                time.sleep(1)
                st.rerun()
            else: st.error("积分不足")