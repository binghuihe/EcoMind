from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from PIL import Image
import os

# ================= 配置区域 (绝对稳健版) =================
DATA_FILE = "garbage_data_train.jsonl" 
MODEL_PATH = "./Qwen2-VL-4bit"
OUTPUT_DIR = "qwen2_vl_garbage_finetune_full"

MAX_SEQ_LENGTH = 1024

# ✅ 核心配置：Batch=1 + 累积16步 = 既省显存又跑得快
BATCH_SIZE = 1 
GRAD_ACCUMULATION = 16 

# ✅ 核心配置：使用 Epoch 策略
# 设为 1 代表把所有数据看 1 遍（通常足够让 LoRA 学会分类规则）
# 如果你觉得效果不够好，可以改成 3
NUM_TRAIN_EPOCHS = 1
# =======================================================

# 1. 加载本地模型
print("🚀 [1/6] 正在加载本地模型...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_PATH,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# 2. 挂载 LoRA 适配器
print("🔗 [2/6] 正在配置 LoRA 参数...")
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3. 加载数据
print(f"📂 [3/6] 正在读取数据: {DATA_FILE}")
dataset = load_dataset("json", data_files = DATA_FILE, split = "train")
print(f"📊 数据集加载完成，共包含 {len(dataset)} 条数据。")

# 4. 自定义数据整理器 (集成 WSL 修复 + 手动 Prompt)
class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            original_messages = example["messages"]
            
            # --- A. 提取信息 ---
            first_content = original_messages[0]["content"]
            image_entry = next((item for item in first_content if item['type'] == 'image'), None)
            text_entry = next((item for item in first_content if item['type'] == 'text'), None)
            # 兼容多轮对话结构，提取 Assistant 的回答
            assist_entry = original_messages[1]["content"][0] if len(original_messages) > 1 else None
            
            if not image_entry or not text_entry or not assist_entry:
                continue

            raw_path = image_entry["image"]
            user_text = text_entry["text"]
            answer_text = assist_entry["text"] if isinstance(assist_entry, dict) else str(assist_entry)
            
            # --- B. WSL 路径自动修复 ---
            image_path = raw_path
            # 如果是 Windows 格式 (C:\...) 且运行在 WSL 环境
            if ":" in raw_path and "\\" in raw_path:
                try:
                    parts = raw_path.split(":", 1)
                    if len(parts) == 2:
                        drive_letter = parts[0].lower()
                        clean_path = parts[1].replace('\\', '/')
                        image_path = f"/mnt/{drive_letter}{clean_path}"
                except:
                    pass
            
            # --- C. 读取图片 ---
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"⚠️ 无法读取图片: {image_path} (已跳过)")
                continue
            
            # --- D. 构造 Prompt (Unsloth 官方推荐格式) ---
            # 手动拼接避免 <|image_pad|> 重复或丢失
            prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n<|im_start|>assistant\n{answer_text}<|im_end|>"
            texts.append(prompt)
            
        if len(images) == 0:
            return None # 遇到坏数据时返回 None，Trainer 会自动跳过

        # --- E. 批量编码 ---
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # 处理标签 (Mask掉 padding 部分)
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

my_collator = Qwen2VLDataCollator(tokenizer)

# 5. 设置训练参数
print("⚙️ [4/6] 配置训练参数...")
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE, 
    gradient_accumulation_steps = GRAD_ACCUMULATION,
    
    # ✅ 关键：按 Epoch 训练，不按 Step
    num_train_epochs = NUM_TRAIN_EPOCHS,
    
    warmup_ratio = 0.05, 
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    
    logging_steps = 10,
    save_strategy = "epoch", # 每跑完一轮保存一次
    
    optim = "adamw_8bit",
    seed = 3407,
    remove_unused_columns = False, 
    label_names = ["labels"],
    report_to = "none", # 不上传 wandb，纯本地
)

# 6. 开始训练
print(f"🔥 [5/6] 开始微调！预计将跑完 {NUM_TRAIN_EPOCHS} 个 Epoch...")
trainer = Trainer(
    model = model,
    train_dataset = dataset,
    data_collator = my_collator,
    args = training_args,
)

trainer_stats = trainer.train()

# 7. 保存结果
print(f"💾 [6/6] 训练完成！正在保存模型到 {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ 所有步骤顺利完成！请使用 inference_rag.py 进行测试。")