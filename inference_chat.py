from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image
import os
import torch

# ================= 配置 =================
BASE_MODEL_PATH = "./Qwen2-VL-4bit"
ADAPTER_PATH = "qwen2_vl_garbage_finetune"
# ========================================

print("🚀 正在初始化模型，请稍候...")
model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
FastVisionModel.for_inference(model)
print("✅ 模型加载完成！")

while True:
    print("\n" + "-"*50)
    # 1. 获取输入
    user_input = input("👉 请输入图片路径 (直接把文件拖进来，输入 q 退出): ").strip()
    
    if user_input.lower() in ['q', 'exit', 'quit']:
        print("👋 Bye!")
        break
        
    # 去除拖拽可能产生的引号
    image_path = user_input.replace('"', '').replace("'", "").strip()
    
    # 2. 路径修复 (WSL)
    if ":" in image_path and "\\" in image_path:
        try:
            drive, rest = os.path.splitdrive(image_path)
            # 暴力字符串替换
            clean_rest = rest.replace('\\', '/')
            image_path = f"/mnt/{drive[0].lower()}{clean_rest}"
        except:
            pass
            
    # 3. 检查文件
    if not os.path.exists(image_path):
        print(f"❌ 找不到文件: {image_path}")
        continue
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ 图片读取失败: {e}")
        continue

    # 4. 推理
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What kind of garbage is this?\n请识别这张图片中的垃圾类别。"}
            ]
        }
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(image, text, add_special_tokens=False, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    final_answer = response.split("assistant")[-1].strip()

    print(f"🤖 识别结果: \033[1;32m{final_answer}\033[0m")