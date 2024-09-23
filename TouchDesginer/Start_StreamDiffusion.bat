
@echo off
cd /d %~dp0
if exist venv (
    PowerShell -Command "& {& 'venv\Scripts\Activate.ps1'; & 'venv\Scripts\python.exe' 'streamdiffusionTD\main_sdtd.py'}"
) else (
    PowerShell -Command "& {& '.venv\Scripts\Activate.ps1'; & '.venv\Scripts\python.exe' 'streamdiffusionTD\main_sdtd.py'}"
)
    pause
            