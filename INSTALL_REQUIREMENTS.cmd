@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Cylinder Solver - install requirements

echo.
echo ==============================================
echo CYLINDER SOLVER - INSTALL REQUIREMENTS
echo ==============================================

set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3"
if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE (
    echo ERROR: Python 3 not found.
    pause
    exit /b 10
)

echo Using: %PYEXE%
%PYEXE% --version

echo.
echo Checking pip...
%PYEXE% -m pip --version
if errorlevel 1 (
    echo pip not found. Running ensurepip...
    %PYEXE% -m ensurepip --upgrade
    if errorlevel 1 goto :fail
)

echo.
echo Installing requirements for the current user...
%PYEXE% -m pip install --user --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

echo.
echo Verifying imports...
%PYEXE% -c "import numpy, matplotlib, xlsxwriter; print('NumPy',numpy.__version__); print('Matplotlib',matplotlib.__version__); print('XlsxWriter',xlsxwriter.__version__)"
if errorlevel 1 goto :fail

echo.
echo INSTALLATION COMPLETED SUCCESSFULLY.
echo Now run START_PROGRAM.cmd
echo.
pause
exit /b 0

:fail
echo.
echo INSTALLATION FAILED.
echo Keep this window open and send a screenshot of the error.
echo.
pause
exit /b 20
