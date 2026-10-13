@echo off
cd /d %~dp0
title DocQA 一键启动

echo ============================================
echo   Milvus-DocQA 一键启动
echo ============================================

echo.
echo [1/4] 检查 Docker 引擎...
docker info >nul 2>&1
if errorlevel 1 (
    echo   Docker 未运行，正在启动 Docker Desktop...
    start "" "C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"
    setlocal enabledelayedexpansion
    set /a tries=0
    :waitdocker
    ping -n 6 127.0.0.1 >nul
    docker info >nul 2>&1
    if errorlevel 1 (
        set /a tries+=1
        if !tries! lss 24 goto waitdocker
        echo   [错误] Docker 启动超时，请手动打开 Docker Desktop 后重试
        endlocal
        pause
        exit /b 1
    )
    endlocal
)
echo   Docker 已就绪

echo.
echo [2/4] 启动 Milvus 向量数据库...
docker compose up -d
echo   等待 Milvus 就绪（首次约需 30 秒）...
:waitmilvus
curl -s -m 2 http://localhost:9091/healthz 2>nul | findstr OK >nul
if errorlevel 1 (
    ping -n 4 127.0.0.1 >nul
    goto waitmilvus
)
echo   Milvus 已就绪

echo.
echo [3/4] 启动后端 API（端口 8000）...
start "DocQA-Backend" ".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000

echo [4/4] 启动问答界面（端口 8501）...
start "DocQA-Frontend" ".venv\Scripts\python.exe" -m streamlit run webui\streamlit_app.py --server.port 8501 --server.headless true

echo.
echo 等待服务就绪后自动打开浏览器...
ping -n 9 127.0.0.1 >nul
start http://localhost:8501

echo.
echo ============================================
echo   全部启动完成！
echo   - 问答界面: http://localhost:8501
echo   - API 文档: http://127.0.0.1:8000/docs
echo   关闭系统请双击 stop.bat
echo ============================================
echo 此窗口可以关闭。
pause >nul


