@echo off
setlocal

set "ROOT=%~dp0.."
set "ROLE=%~1"
set "MODE=%~2"

if "%ROLE%"=="" set "ROLE=analyst"

pushd "%ROOT%"
set "PYTHONPATH=%ROOT%\src;%PYTHONPATH%"

if /I "%MODE%"=="--debug" (
    echo Starting desktop in debug mode on 127.0.0.1:5678, role=%ROLE%
    python -m debugpy --listen 127.0.0.1:5678 -m fire_es_desktop.main --role %ROLE%
) else (
    echo Starting desktop from sources, role=%ROLE%
    python -m fire_es_desktop.main --role %ROLE%
)

popd
endlocal
