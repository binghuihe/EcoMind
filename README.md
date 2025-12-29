# ♻️ EcoMind 城市暗夜守护者: 基于多模态大模型的城市废弃物全链路管理系统

> **课程名称**：创新创业实践  
> **提交人**：[你的名字]  
> **提交日期**：2025年X月X日

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Model](https://img.shields.io/badge/Model-Qwen2--VL-violet)
![Framework](https://img.shields.io/badge/Inference-Unsloth-green)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange)

---

## 📖 1. 项目简介 (Project Introduction)

**EcoMind** 是一款基于**多模态大语言模型 (Multimodal LLM)** 与 **检索增强生成 (RAG)** 技术的城市级废弃物智能管理终端。

[cite_start]针对当前垃圾分类中“分不清、不想分、管不住”的痛点，本项目通过微调 **Qwen2-VL-7B** 视觉大模型，实现对非标品、破损垃圾的精准语义识别 [cite: 7, 12][cite_start]。同时，系统接入本地向量知识库，能像环保专家一样与用户进行多轮对话，提供有法律依据的投放指导 [cite: 7, 14]。

**核心创新点：**
* [cite_start]**准**：基于 Vision Transformer 架构，解决复杂垃圾识别难题 [cite: 12]。
* [cite_start]**专**：独创“看-查-说”三步推理机制，结合 RAG 技术杜绝大模型幻觉 [cite: 16]。
* [cite_start]**快**：采用 Unsloth 4-bit 量化技术，在边缘侧显卡（RTX 4070 Super）上实现毫秒级响应 。

---

## 📸 2. 项目演示 (Project Demo)

> 以下为项目在本地 WSL 环境下的实际运行截图：

### 2.1 核心功能：AI 智能识别与专家对话
左图展示了系统识别出复杂垃圾（如快递包装）后，不仅给出了分类建议（蓝色-可回收），还通过 RAG 技术回答了用户关于“能卖钱吗”的追问，引用了相关回收标准。

![智能识别](000034.jpg)

### 2.2 数据可视化：城市态势驾驶舱
实时监控区域内的垃圾吞吐量、资源化利用率及成分光谱，为管理者提供决策支持。

![数据驾驶舱](000048.jpg)

### 2.3 更多功能
| 赛博回收地图 (LBS) | 碳普惠积分商城 |
|:---:|:---:|
| ![地图](000079.jpg) | ![商城](000096.jpg) |
| [cite_start]*基于 GIS 的设施定位* [cite: 31] | [cite_start]*游戏化积分兑换系统* [cite: 36] |

---

## 📂 3. 项目文件结构 (File Structure)

本项目采用模块化设计，主要文件说明如下：

```text
EcoMind/
[cite_start]├── app.py                  # 🚀 [核心入口] Streamlit 前端主程序，包含 UI 布局与交互逻辑 [cite: 9, 15]
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