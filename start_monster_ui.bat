@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%webui"

set "LOCAL_UV_VENV=%PROJECT_DIR%.venv-uv\Scripts\python.exe"
set "LOCAL_VENV=%PROJECT_DIR%.venv\Scripts\python.exe"
set "PARENT_UV_VENV=%PROJECT_DIR%..\.venv-uv\Scripts\python.exe"
set "PARENT_VENV=%PROJECT_DIR%..\.venv\Scripts\python.exe"

if exist "%LOCAL_UV_VENV%" (
    set "PYTHON_EXE=%LOCAL_UV_VENV%"
) else if exist "%LOCAL_VENV%" (
    set "PYTHON_EXE=%LOCAL_VENV%"
) else if exist "%PARENT_UV_VENV%" (
    set "PYTHON_EXE=%PARENT_UV_VENV%"
) else if exist "%PARENT_VENV%" (
    set "PYTHON_EXE=%PARENT_VENV%"
) else (
    set "PYTHON_EXE=python"
)

echo Starting Kronos Monster 2.0 UI...
echo Using Python: %PYTHON_EXE%
echo URL: http://127.0.0.1:7070
echo.

"%PYTHON_EXE%" serve.py

echo.
echo Server stopped. If there was an error, review the message above.
pause
