
            @echo off
            echo Current directory: %CD%
            cd /d "C:/Users/osrin/Downloads/BASE0715/StreamDiffusion"
            echo Changed directory to: %CD%
            set "PIP_DISABLE_PIP_VERSION_CHECK=1"
            if not exist "venv" (
                echo Creating Python venv at: "C:/Users/osrin/Downloads/BASE0715/StreamDiffusion\venv"
                "C:/Users/osrin/AppData/Local/Programs/Python/Python310/python.exe" -m venv venv
            ) else (
                echo Virtual environment already exists at: "C:/Users/osrin/Downloads/BASE0715/StreamDiffusion\venv"
            )

            echo Attempting to activate virtual environment...
            call "venv\Scripts\activate.bat"

            rem Check if the virtual environment was activated successfully
            if "%VIRTUAL_ENV%" == "" (
                echo Failed to activate virtual environment. Please check the path and ensure the venv exists.
                echo Path to venv: "C:/Users/osrin/Downloads/BASE0715/StreamDiffusion\venv"
                echo VIRTUAL_ENV: "%VIRTUAL_ENV%"
                pause /b 1
            ) else (
                echo Virtual environment activated.
            )

            echo Installing 'wheel' to ensure successful building of packages...
            python -m pip install wheel

            echo Installing nvidia-pyindex to ensure access to NVIDIA-specific packages...
            python -m pip install nvidia-pyindex

            echo Installing dependencies with pip from the activated virtual environment...
            python -m pip install --upgrade pip
            python -m pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
            python -m pip install -e .
            python setup.py develop
            python -m pip install -r streamdiffusionTD/requirements.txt

            echo Installation Finished
            pause
        