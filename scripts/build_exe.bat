@echo off
setlocal

REM scripts/build_exe.bat
REM Build Fire ES Desktop into dist\FireES\FireES.exe

set "ROOT=%~dp0.."
set "SPEC_FILE=%ROOT%\build\pyinstaller.spec"
set "WORK_PATH=%ROOT%\build\pyinstaller"
set "DIST_PATH=%ROOT%\dist"

echo ========================================
echo Fire ES Desktop - Build EXE
echo ========================================
echo.

if not exist "%SPEC_FILE%" (
    echo [ERROR] Missing spec file: %SPEC_FILE%
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH.
    exit /b 1
)

pushd "%ROOT%"

echo [1/4] Installing project dependencies...
python -m pip install -e . >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install project dependencies.
    popd
    exit /b 1
)

echo [2/4] Installing build dependencies...
python -m pip install pyinstaller PySide6 >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install build dependencies.
    popd
    exit /b 1
)

echo [3/4] Cleaning previous build output...
if exist "%WORK_PATH%" rmdir /s /q "%WORK_PATH%"
if exist "%DIST_PATH%\FireES" rmdir /s /q "%DIST_PATH%\FireES"

echo [4/4] Running PyInstaller...
python -m PyInstaller "%SPEC_FILE%" --clean --workpath "%WORK_PATH%" --distpath "%DIST_PATH%" -y
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Build failed.
    echo ========================================
    popd
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully.
echo ========================================
echo Artifact: dist\FireES\FireES.exe
echo Run:
echo   dist\FireES\FireES.exe --role analyst
echo   dist\FireES\FireES.exe --role lpr
echo.

popd
endlocal
