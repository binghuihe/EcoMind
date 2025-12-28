from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image
import os
import torch

# ================= 配置区域 =================
BASE_MODEL_PATH = "./Qwen2-VL-4bit"
ADAPTER_PATH = "qwen2_vl_garbage_finetune"

# 👇 这里的路径不需要改，下面的代码会自动修
TEST_IMAGE_PATH = r"C:\Users\Administrator\Desktop\garbage_classification\battery\battery10.jpg"
# ===========================================

print("🚀 [1/3] 正在加载本地基础模型 (绝对不联网)...")
model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

print(f"🔗 [2/3] 正在挂载 LoRA 适配器: {ADAPTER_PATH} ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
FastVisionModel.for_inference(model) 

# ================= 路径修复 (暴力字符串版) =================
image_path = TEST_IMAGE_PATH

# 只要包含 : 和 \ 就认为是 Windows 路径，强制转换
if ":" in image_path and "\\" in image_path:
    try:
        # 手动切割，不依赖 os.path.splitdrive
        # "C:\Users..." -> ["C", "\Users..."]
        parts = image_path.split(":", 1)
        
        if len(parts) == 2:
            drive_letter = parts[0].lower() # 拿到 "c"
            rest_of_path = parts[1]         # 拿到 "\Users..."
            
            # 替换反斜杠
            clean_rest = rest_of_path.replace('\\', '/')
            
            # 拼装成 /mnt/c/Users...
            image_path = f"/mnt/{drive_letter}{clean_rest}"
            
    except Exception as e:
        print(f"⚠️ 路径转换出错: {e}")

print(f"🖼️ [3/3] 正在测试图片: {image_path}")

try:
    image = Image.open(image_path).convert("RGB")
    print("✅ 图片读取成功！")
except Exception as e:
    print(f"❌ 无法读取图片，请检查路径。\n错误: {e}")
    exit()

# ================= 构造 Prompt 并推理 =================
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

inputs = tokenizer(
    image,
    text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("🤖 模型正在思考...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    use_cache=True,
    temperature=0.1, 
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
final_answer = response.split("assistant")[-1].strip()

print("\n" + "="*30)
print(f"🔮 预测结果: {final_answer}")
print("="*30)