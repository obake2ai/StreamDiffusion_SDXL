
@echo off
echo Current directory: %CD%
cd /d "C:/Users/osrin/Downloads/BASE0715/StreamDiffusion"

echo Attempting to activate virtual environment...
call "venv\Scripts\activate.bat"

rem Check if the virtual environment was activated successfully
if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment. Please check the path and ensure the venv exists.
    pause /b 1
) else (
    echo Virtual environment activated.
)

echo Installing TensorRT...
python -m streamdiffusion.tools.install-tensorrt

echo TensorRT installation finished
pause
        