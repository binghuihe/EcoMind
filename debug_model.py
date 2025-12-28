import torch
from unsloth import FastVisionModel
from PIL import Image
import os

# 1. 模拟加载模型
print("🔄 [1/4] 正在加载模型...")
model, tokenizer = FastVisionModel.from_pretrained(
    "./Qwen2-VL-4bit",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# 2. 模拟加载图片 (自动找一张你上传过的图片，或者由你指定)
# 请确保你的目录下有一张 jpg 图片，这里我写死一个名字，你需要改成你真实存在的图片名
image_path = '/home/hui/ocr_gb/wx_chatou1.jpg'
# ▲▲▲ 注意：如果没有 test.jpg，代码会报错，请把这里改成你文件夹里随便一张垃圾图片的名字 ▲▲▲

if not os.path.exists(image_path):
    # 尝试自动找一张
    files = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    if files:
        image_path = files[0]
        print(f"⚠️ 未找到指定图片，自动使用: {image_path}")
    else:
        print("❌ 错误：当前目录下没有图片，无法测试！")
        exit()

print(f"📸 [2/4] 正在读取图片: {image_path}")
image = Image.open(image_path).convert("RGB")

# 3. 构造最简单的 Prompt
print("🧠 [3/4] 正在尝试推理...")
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "详细描述一下这张图片，并告诉我它属于什么垃圾？"}
    ]}
]

text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(image, text_prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

# 4. 生成 (强制打印每一个 token)
print("🚀 [4/4] 开始生成 (请盯着下面)...")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=False # 使用贪婪解码，最稳定
    )

# 5. 解码
print("-" * 30)
print("RAW OUTPUT (原始 Token ID):")
print(outputs)
print("-" * 30)
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False) # 不跳过特殊字符，看看有没有 <|endoftext|>
print("DECODED TEXT (解码文本):")
print(decoded_text)
print("-" * 30)

if "assistant" in decoded_text:
    print("✅ 成功！提取出的回答：")
    print(decoded_text.split("assistant")[-1])
else:
    print("❌ 失败！模型输出了内容，但格式不对 (找不到 'assistant' 标记)。")