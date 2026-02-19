@echo off
REM scripts/build_exe.bat
REM Скрипт сборки Fire ES Desktop в .exe

set "SPEC_FILE=build\pyinstaller.spec"

echo ========================================
echo Fire ES Desktop — Сборка .exe
echo ========================================
echo.

REM Проверка spec-файла
if not exist "%SPEC_FILE%" (
    echo Ошибка: не найден spec-файл %SPEC_FILE%
    exit /b 1
)

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден в PATH
    exit /b 1
)

REM Установка зависимостей
echo [1/4] Установка зависимостей...
pip install -e . >nul 2>&1
if errorlevel 1 (
    echo Ошибка установки зависимостей
    exit /b 1
)

REM Установка PyInstaller
echo [2/4] Установка PyInstaller...
pip install pyinstaller PySide6 >nul 2>&1
if errorlevel 1 (
    echo Ошибка установки PyInstaller
    exit /b 1
)

REM Очистка предыдущей сборки
echo [3/4] Очистка предыдущей сборки...
if exist "build\pyinstaller" rmdir /s /q "build\pyinstaller"
if exist "dist\FireES" rmdir /s /q "dist\FireES"

REM Сборка
echo [4/4] Сборка .exe...
pyinstaller "%SPEC_FILE%" --clean

if errorlevel 1 (
    echo.
    echo ========================================
    echo Ошибка сборки!
    echo ========================================
    exit /b 1
)

echo.
echo ========================================
echo Сборка завершена успешно!
echo ========================================
echo.
echo Исполняемый файл: dist\FireES\FireES.exe
echo.
echo Запуск:
echo   dist\FireES\FireES.exe --role analyst
echo   dist\FireES\FireES.exe --role lpr
echo.
