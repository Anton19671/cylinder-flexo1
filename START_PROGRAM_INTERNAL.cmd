@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Cylinder Solver - startup

set "LOG=%~dp0STARTUP_LOG.txt"
> "%LOG%" echo ==================================================
>>"%LOG%" echo CYLINDER SOLVER STARTUP
>>"%LOG%" echo DATE/TIME: %date% %time%
>>"%LOG%" echo FOLDER: %CD%
>>"%LOG%" echo ==================================================

echo.
echo ==============================================
echo CYLINDER SOLVER
echo ==============================================
echo Checking Python...

set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3"

if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE goto :no_python

echo Python command: %PYEXE%
>>"%LOG%" echo PYTHON COMMAND: %PYEXE%
%PYEXE% --version
%PYEXE% --version >>"%LOG%" 2>&1
if errorlevel 1 goto :python_error

echo.
echo Checking tkinter...
%PYEXE% -c "import tkinter; print('tkinter OK')" >>"%LOG%" 2>&1
if errorlevel 1 goto :tk_error
echo tkinter: OK

echo.
echo Checking pip...
%PYEXE% -m pip --version >>"%LOG%" 2>&1
if errorlevel 1 (
    echo pip is missing. Trying ensurepip...
    %PYEXE% -m ensurepip --upgrade
    %PYEXE% -m ensurepip --upgrade >>"%LOG%" 2>&1
    if errorlevel 1 goto :pip_error
)

echo.
echo Checking NumPy, Matplotlib, XlsxWriter and VIDEO encoder...
%PYEXE% -c "import numpy, scipy, matplotlib, xlsxwriter, imageio_ffmpeg, yaml; print('numpy',numpy.__version__); print('matplotlib',matplotlib.__version__); print('xlsxwriter',xlsxwriter.__version__); print('imageio_ffmpeg OK')" >>"%LOG%" 2>&1
if not errorlevel 1 goto :packages_ok

echo Required Python packages are missing.
echo Installing required Python packages for the current user...
echo This may take several minutes and requires Internet access.
echo.

%PYEXE% -m pip install --user --disable-pip-version-check -r "%~dp0requirements.txt" >>"%LOG%" 2>&1
set "PIP_RC=%ERRORLEVEL%"

if not "%PIP_RC%"=="0" goto :install_error

echo.
echo Re-checking installed packages...
%PYEXE% -c "import numpy, matplotlib, xlsxwriter, imageio_ffmpeg; print('numpy',numpy.__version__); print('matplotlib',matplotlib.__version__); print('xlsxwriter',xlsxwriter.__version__); print('imageio_ffmpeg OK')"
%PYEXE% -c "import numpy, matplotlib, xlsxwriter, imageio_ffmpeg; print('numpy',numpy.__version__); print('matplotlib',matplotlib.__version__); print('xlsxwriter',xlsxwriter.__version__); print('imageio_ffmpeg OK')" >>"%LOG%" 2>&1
if errorlevel 1 goto :install_import_error

:packages_ok
echo Packages: OK

echo.
echo Checking solver GUI import...
%PYEXE% -c "import FINAL_GUI; print('FINAL_GUI import OK')" >>"%LOG%" 2>&1
if errorlevel 1 goto :gui_import_error
echo Solver import: OK

echo.
echo Launching GUI as an independent window...
echo.
>>"%LOG%" echo Resolving independent GUI interpreter...

set "GUI_PY_FILE=%TEMP%\cylinder_gui_python_%RANDOM%_%RANDOM%.txt"
%PYEXE% -c "import pathlib,sys; p=pathlib.Path(sys.executable); w=p.with_name('pythonw.exe'); print(str(w if w.exists() else p))" > "%GUI_PY_FILE%" 2>>"%LOG%"
if errorlevel 1 goto :gui_launch_error
set /p "GUI_PY="<"%GUI_PY_FILE%"
del /q "%GUI_PY_FILE%" >nul 2>&1
if not defined GUI_PY goto :gui_launch_error

>>"%LOG%" echo GUI INTERPRETER: %GUI_PY%
>>"%LOG%" echo GUI BOOTSTRAP: %~dp0GUI_BOOTSTRAP.pyw

rem START creates a separate process. pythonw.exe has no console and is not tied to this CMD window.
start "" /D "%~dp0" "%GUI_PY%" "%~dp0GUI_BOOTSTRAP.pyw"
if errorlevel 1 goto :gui_launch_error

echo GUI started independently.
echo This console window can now be closed safely.
>>"%LOG%" echo GUI started independently.
timeout /t 2 /nobreak >nul 2>&1
exit /b 0

:gui_launch_error
echo.
echo ERROR: independent GUI launch failed.
echo See STARTUP_LOG.txt in this folder.
echo.
>>"%LOG%" echo ERROR: independent GUI launch failed.
pause
exit /b 17

:no_python
echo.
echo ERROR: Python 3 was not found.
echo Install Python 3 for Windows from python.org.
echo During installation enable "Add python.exe to PATH".
echo Then run START_PROGRAM.cmd again.
>>"%LOG%" echo ERROR: Python 3 not found.
pause
exit /b 10

:python_error
echo.
echo ERROR: Python was found but cannot be started.
echo See STARTUP_LOG.txt.
pause
exit /b 11

:tk_error
echo.
echo ERROR: tkinter is not available in this Python installation.
echo Reinstall the standard Windows Python distribution with Tcl/Tk support.
echo See STARTUP_LOG.txt.
pause
exit /b 12

:pip_error
echo.
echo ERROR: pip is not available and ensurepip failed.
echo See STARTUP_LOG.txt.
pause
exit /b 13

:install_error
echo.
echo ERROR: automatic package installation failed.
echo Check Internet access and STARTUP_LOG.txt.
echo.
pause
exit /b 14

:install_import_error
echo.
echo ERROR: packages were installed but Python still cannot import them.
echo See STARTUP_LOG.txt.
echo.
pause
exit /b 15

:gui_import_error
echo.
echo ERROR: solver GUI import failed.
echo See STARTUP_LOG.txt.
echo.
pause
exit /b 16
