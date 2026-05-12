@echo off
echo ============================================
echo  RL Signal Routing - Environment Setup
echo ============================================

REM ---- STEP 1: Check for Python 3.11, install if missing ----
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Python 3.11 not found. Downloading installer...
    bitsadmin /transfer "PythonDownload" ^
        https://www.python.org/ftp/python/3.11.10/python-3.11.10-amd64.exe ^
        "%TEMP%\python-3.11.10-amd64.exe"
    echo [INFO] Installing Python 3.11 silently...
    "%TEMP%\python-3.11.10-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del "%TEMP%\python-3.11.10-amd64.exe"
    echo [OK] Python 3.11 installed
) else (
    echo [OK] Python 3.11 already installed
)

REM Refresh PATH so py launcher sees the new install
call refreshenv >nul 2>&1

REM ---- STEP 2: Create venv if missing ----
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    py -3.13 -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM ---- STEP 3: Install dependencies if missing ----
venv\Scripts\python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies - this may take 10-15 minutes...
    venv\Scripts\pip install --upgrade pip
    venv\Scripts\pip install tensorflow==2.18.0 --timeout 300
    venv\Scripts\pip install "numpy>=1.26.4,<2.0" pandas scipy seaborn matplotlib
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM ---- STEP 4: Check SUMO ----
where sumo >nul 2>&1
if errorlevel 1 (
    echo [WARNING] SUMO not found in PATH.
    echo Please install SUMO from https://sumo.dlr.de/docs/Downloads.php
    echo and set the SUMO_HOME environment variable.
    pause
    exit /b 1
)
echo [OK] SUMO found

REM ---- STEP 5: Check malta.net.xml ----
if not exist "Malta\data\malta.net.xml" (
    echo [WARNING] Malta\data\malta.net.xml is missing!
    pause
    exit /b 1
)
echo [OK] malta.net.xml found

echo.
echo ============================================
echo  Setup complete! Starting training...
echo ============================================
echo.
venv\Scripts\python main.py