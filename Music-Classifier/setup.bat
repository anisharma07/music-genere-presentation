@echo off
REM Setup script for Music Genre Classifier (Windows)

echo ==========================================
echo Music Genre Classifier - Setup Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo V Python found

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ==========================================
echo V Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Train models: python train_models.py
echo 3. Start server: python app.py
echo 4. Open browser: http://localhost:5000
echo.
echo ==========================================
pause
