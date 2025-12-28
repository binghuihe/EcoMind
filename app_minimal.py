import streamlit as st
from unsloth import FastVisionModel
from PIL import Image
import torch

st.title("🛠️ 极简 Debug 模式")

# 1. 加载模型
@st.cache_resource
def load_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "./Qwen2-VL-4bit",
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

try:
    model, tokenizer = load_model()
    st.success("模型加载成功")
except Exception as e:
    st.error(f"模型挂了: {e}")
    st.stop()

# 2. 上传
uploaded_file = st.file_uploader("传张图试试", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    # 3. 聊天框 (直接测试生成)
    user_input = st.text_input("问点什么 (例如: 这是什么?)", "这是什么?")
    
    if st.button("发送测试"):
        # 构造 Prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_input}
            ]}
        ]
        
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(image, text_prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

        # 显示调试信息
        st.write("正在生成 Token...")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
            
        # 暴力显示所有输出，不做 split
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.write("--- 原始输出 ---")
        st.code(raw_text) # 把原本模型吐出来的所有东西都显示出来
        
        st.write("--- 尝试提取 ---")
        if "assistant" in raw_text:
            st.success(raw_text.split("assistant")[-1])
        else:
            st.warning("没找到 assistant 标记，直接显示最后部分:")
            st.info(raw_text)