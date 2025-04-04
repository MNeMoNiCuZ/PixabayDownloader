@echo off
cd /d "%~dp0"

:: Check if we're already in a virtual environment
if defined VIRTUAL_ENV (
    :: Use the virtual environment's Python
    "%VIRTUAL_ENV%\Scripts\python.exe" Scripts\gui.py
) else (
    :: Try to activate the virtual environment first
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
        venv\Scripts\python.exe scripts\gui.py
    ) else (
        :: Fallback to system Python
        python Scripts\gui.py
    )
)

if errorlevel 1 (
    echo Failed to launch application. Please check that Python is installed and all dependencies are installed.
    echo Try running: pip install -r requirements.txt
    pause
)