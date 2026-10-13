@echo off
cd /d %~dp0
title DocQA 一键停止

echo ============================================
echo   Milvus-DocQA 一键停止
echo ============================================

echo.
echo [1/2] 停止问答界面和后端 API...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
echo   已停止

echo.
echo [2/2] 停止 Milvus 数据库（数据保留）...
docker compose stop
echo   已停止

echo.
echo ============================================
echo   全部已停止（文档数据已保留）
echo   下次使用双击 start.bat
echo ============================================
pause

