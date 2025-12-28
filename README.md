# ♻️ EcoMind: 基于多模态大模型的城市废弃物全链路管理系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Model](https://img.shields.io/badge/Model-Qwen2--VL-violet)
![Framework](https://img.shields.io/badge/Inference-Unsloth-green)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange)

## 📖 项目介绍 (Introduction)

[cite_start]**EcoMind** 是一款基于**多模态大语言模型 (Multimodal LLM)** 与 **检索增强生成 (RAG)** 技术的城市级废弃物智能管理终端 [cite: 7]。

[cite_start]针对当前垃圾分类中“分不清、不想分、管不住”的痛点 [cite: 3, 4, 5][cite_start]，EcoMind 被定义为“拥有环保专家大脑的智能体”。它不仅能通过视觉“看懂”垃圾，还能通过接入本地知识库，像专家一样与用户对话，提供从投放指导到科普教育的一站式服务 [cite: 7]。

**核心价值：**
* [cite_start]**准**：利用 7B 参数量的视觉大模型，实现对非标品、破损垃圾的精准语义识别 [cite: 7]。
* [cite_start]**专**：外挂工业级处置知识库，确保每一次回答都有据可查，杜绝大模型“幻觉” [cite: 7]。
* [cite_start]**快**：采用边缘侧量化部署方案，在消费级显卡（RTX 4070 Super）上实现毫秒级响应 [cite: 7, 13]。

## 🏗️ 系统架构 (Architecture)

[cite_start]本项目采用“端-边-云”协同架构，前端基于 Streamlit，后端集成 Unsloth 加速推理引擎与 ChromaDB 向量数据库 [cite: 9]。

### 核心逻辑层
1.  [cite_start]**感知层**：支持摄像头实时采集与图像上传 [cite: 10]。
2.  **认知层**：
    * [cite_start]**视觉基座**：Qwen2-VL-7B-Instruct [cite: 12]。
    * [cite_start]**推理加速**：Unsloth 4-bit QLoRA 微调 [cite: 13]。
    * [cite_start]**RAG 知识库**：ChromaDB + all-MiniLM-L6-v2 。
3.  [cite_start]**应用层**：Streamlit + Plotly + Folium [cite: 15]。

### 技术路线图
```mermaid
graph TD
    User[用户终端] -->|上传图像/文本| App[Streamlit 前端]
    
    subgraph "核心计算层 (Edge Server)"
        App -->|预处理| Vision[视觉编码器: Qwen2-VL]
        Vision -->|提取特征| LoRA[微调适配器: Unsloth QLoRA]
        LoRA -->|生成类别| LLM[推理引擎]
        
        LLM -->|查询 Query| VectorDB[(向量数据库: ChromaDB)]
        VectorDB -->|检索 RAG Evidence| LLM
    end
    
    LLM -->|生成最终建议| App
    App -->|可视化渲染| Display[结果展示: 卡片/对话/地图]