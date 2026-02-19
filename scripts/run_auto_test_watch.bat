@echo off
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"

python scripts\auto_test_watch.py ^
  --watch ^
  --max-iterations 20 ^
  --max-hours 6 ^
  --strict-green ^
  --commit-each-green ^
  --reports-dir reports/auto_test_runs ^
  %*

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

