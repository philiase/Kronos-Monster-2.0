@echo off
setlocal

cd /d "%~dp0webui"

set "LOCAL_VENV=%~dp0.venv\Scripts\python.exe"

if exist "%LOCAL_VENV%" (
    set "PYTHON_EXE=%LOCAL_VENV%"
) else (
    set "PYTHON_EXE=python"
)

echo Starting Kronos Monster 2.0 UI...
echo Using Python: %PYTHON_EXE%
echo URL: http://127.0.0.1:7070
echo.

"%PYTHON_EXE%" serve.py

echo.
echo Server stopped. If there was an error, send the message above to Chat.
pause
