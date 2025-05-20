@echo off
setlocal enabledelayedexpansion

set PYTHON_URL=https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip
set PIP_URL=https://bootstrap.pypa.io/get-pip.py
set HF_ENDPOINT=https://hf-mirror.com
set PIP_MIRROR=https://pypi.org/simple
set PYTHON_MIRROR=https://www.python.org/ftp/python/3.12.0/python-3.12.0-embed-amd64.zip

if not exist python.zip (
    powershell -Command "& {Invoke-WebRequest -Uri %PYTHON_MIRROR% -OutFile python.zip}"
)
if not exist drpdf_dist/python.exe (
    mkdir drpdf_dist
    powershell -Command "& {Expand-Archive -Path python.zip -DestinationPath drpdf_dist -Force}"
    echo python312.zip >> drpdf_dist/python312._pth
    echo import site >> drpdf_dist/python312._pth
)

cd drpdf_dist
if not exist get-pip.py (
    powershell -Command "& {Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py}"
)
if not exist Scripts/pip.exe (
    python get-pip.py --no-warn-script-location
)

pip install --no-warn-script-location --upgrade drpdf -i !PIP_MIRROR!
drpdf -i

pause
