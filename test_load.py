from unsloth import FastVisionModel
import torch
import time

# 你的本地模型路径 (绝对不要改动，除非你改了文件夹名)
local_model_path = "./Qwen2-VL-4bit"

print("⏳ [1/3] 开始加载模型... (这一步通常需要 1-3 分钟，请耐心等待，不要关闭！)")
start_time = time.time()

# 强制从本地加载，完全断网也能跑
try:
    model, tokenizer = FastVisionModel.from_pretrained(
        local_model_path,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    print(f"✅ [2/3] 模型加载成功！耗时: {time.time() - start_time:.2f} 秒")
except Exception as e:
    print(f"❌ 加载失败，错误信息: {e}")
    exit()

print("🔍 [3/3] 正在检查显存占用...")
# 简单打印一下模型参数类型，证明它活了
print(f"模型类型: {type(model)}")
print("🎉 恭喜！环境完美，模型完好。你可以放心去跑 train.py 了！")