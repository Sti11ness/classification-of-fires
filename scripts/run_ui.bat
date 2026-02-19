@echo off
echo ========================================
echo Fire ES - Запуск Streamlit UI
echo ========================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден!
    exit /b 1
)

REM Проверка streamlit
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit не установлен!
    echo Установка...
    pip install streamlit
)

echo.
echo [INFO] Запуск Streamlit...
echo.
echo URL: http://localhost:8501
echo.

REM Запуск streamlit
streamlit run app\streamlit_app.py --server.port 8501 --server.headless true

pause
