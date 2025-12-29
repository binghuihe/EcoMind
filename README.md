Markdown# ♻️ EcoMind: 基于多模态大模型的城市废弃物全链路管理系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Model](https://img.shields.io/badge/Model-Qwen2--VL-violet)
![Framework](https://img.shields.io/badge/Inference-Unsloth-green)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange)

---

## 📖 1. 项目简介 (Project Introduction)

[cite_start]**EcoMind** 是一款基于**多模态大语言模型 (Multimodal LLM)** 与 **检索增强生成 (RAG)** 技术的城市级废弃物智能管理终端 [cite: 7, 18]。

[cite_start]针对当前垃圾分类中“分不清、不想分、管不住”的痛点，本项目通过微调 **Qwen2-VL-7B** 视觉大模型，实现对非标品、破损垃圾的精准语义识别 [cite: 3, 12][cite_start]。同时，系统接入本地向量知识库，能像环保专家一样与用户进行多轮对话，提供有法律依据的投放指导 [cite: 14, 22]。

**核心创新点：**
* [cite_start]**准**：基于 Vision Transformer 架构，解决复杂垃圾识别难题 [cite: 12]。
* [cite_start]**专**：独创“看-查-说”三步推理机制，结合 RAG 技术杜绝大模型幻觉 [cite: 16]。
* [cite_start]**快**：采用 Unsloth 4-bit 量化技术，在边缘侧显卡（RTX 4070 Super）上实现毫秒级响应 [cite: 13]。

---

## ⚙️ 2. 环境配置与安装 (Environment Setup)

> 本项目在 **Windows Subsystem for Linux (WSL2)** 环境下开发，硬件环境为 NVIDIA RTX 4070 Super (12GB)。

### 2.1 基础环境要求
* **操作系统**: Ubuntu 20.04 / 22.04 (WSL2)
* **Python**: 3.10+
* **CUDA**: 12.1 (用于 GPU 加速)
* **显存**: 建议 >= 8GB (运行 4-bit 量化模型)

### 2.2 安装步骤 (Installation)

**Step 1: 克隆项目**
```bash
git clone [https://github.com/binghuihe/EcoMind.git](https://github.com/binghuihe/EcoMind.git)
cd EcoMind
Step 2: 创建虚拟环境Bash# 建议使用 Conda 管理环境
conda create -n ecomind python=3.10
conda activate ecomind
Step 3: 安装核心依赖Bash# 1. 安装 PyTorch 与 CUDA 支持 (对应 CUDA 12.1)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. 安装 Unsloth (用于大模型推理加速)
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"

# 3. 安装项目依赖 (Streamlit, ChromaDB, Plotly 等)
pip install -r requirements.txt
📸 3. 项目演示 (Project Demo)以下为项目在本地 WSL 环境下的实际运行效果。3.1 核心功能：AI 智能识别与专家对话（左图：识别出复杂垃圾并给出分类建议；右图：通过 RAG 技术回答用户追问）识别界面对话详情3.2 数据可视化：城市态势驾驶舱实时监控区域内的垃圾吞吐量、资源化利用率及成分光谱。3.3 更多功能赛博回收地图 (LBS)碳普惠积分商城📂 4. 项目文件结构 (File Structure)本项目采用模块化设计，主要文件说明如下：PlaintextEcoMind/
├── app.py                  # 🚀 [核心入口] Streamlit 前端主程序，包含 UI 布局与交互逻辑
├── start.sh                # 🛠️ [启动脚本] 自动化启动服务并处理端口转发
├── requirements.txt        # 📦 [依赖清单] 项目所需的 Python 库列表
├── .gitignore              # ⚙️ [Git配置] 忽略大模型权重与临时文件
│
├── Qwen2-VL-4bit/          # 🧠 [模型权重] 经过 Unsloth 量化微调后的视觉大模型 (本地加载)
│   ├── adapter_config.json
│   └── ... (模型分片文件)
│
[cite_start]├── rag_db/                 # 🗄️ [知识库] ChromaDB 向量数据库文件 [cite: 14]
│   ├── chroma.sqlite3      # 向量索引数据
│   └── ...
│
├── .streamlit/             # 🎨 [UI配置] Streamlit 的主题与页面设置
│   └── config.toml
│
└── README.md               # 📄 项目说明文档
🚀 5. 快速启动 (Usage)在终端运行提供的启动脚本即可一键部署：Bash# 赋予脚本执行权限
chmod +x start.sh

# 启动系统
./start.sh
启动成功后，访问终端显示的本地链接 (如 http://localhost:8501) 即可使用。