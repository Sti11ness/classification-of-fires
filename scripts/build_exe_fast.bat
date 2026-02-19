@echo off
setlocal

set "ROOT=%~dp0.."
set "SPEC_FILE=%ROOT%\build\pyinstaller.spec"
set "WORK_PATH=%ROOT%\build\pyinstaller_fast"
set "DIST_PATH=%ROOT%\dist_new"

if not exist "%SPEC_FILE%" (
    echo [ERROR] Missing spec file: %SPEC_FILE%
    exit /b 1
)

pushd "%ROOT%"
echo Building FireES.exe to dist_new using incremental cache...
pyinstaller "%SPEC_FILE%" --workpath "%WORK_PATH%" --distpath "%DIST_PATH%" -y
if errorlevel 1 (
    echo [ERROR] Build failed.
    popd
    exit /b 1
)

echo [OK] Artifact: dist_new\FireES\FireES.exe
popd
endlocal
