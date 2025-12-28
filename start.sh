#!/bin/bash

# ================= 配置区域 =================
APP_FILE="app.py"
LOG_FILE="run.log"
PORT=8501
# ===========================================

echo "🟢 [1/4] 正在后台启动 Streamlit 服务..."
# 1. 后台启动 Streamlit
nohup streamlit run $APP_FILE > $LOG_FILE 2>&1 &
STREAMLIT_PID=$!
echo "   ✅ 服务已启动 (PID: $STREAMLIT_PID)"

# 定义清理函数：按 Ctrl+C 时自动关闭后台服务
cleanup() {
    echo ""
    echo "🔴 检测到退出信号，正在关闭 Streamlit 服务..."
    kill $STREAMLIT_PID
    echo "👋 服务已关闭，再见！"
    exit
}
trap cleanup SIGINT

echo "⏳ [2/4] 等待 10 秒让服务初始化..."
sleep 10

echo "🔥 [3/4] 发送预热请求 (加载模型到显存)..."
curl -s -o /dev/null http://localhost:$PORT
echo "   ✅ 预热信号已发送！模型正在加载中..."

echo "🌍 [4/4] 正在建立公网穿透 (Serveo)..."
echo "-----------------------------------------------------"
echo "👇 请复制下方出现的绿色 https 链接 (发给手机/Mac) 👇"
echo "💡 提示：演示结束后，在这个窗口按 Ctrl+C 即可一键关闭所有服务"
echo "-----------------------------------------------------"

# 2. 前台启动 SSH 穿透 (关键步骤！)
# 使用 -o ServerAliveInterval=60 防止断连
ssh -o ServerAliveInterval=60 -R 80:localhost:$PORT serveo.net