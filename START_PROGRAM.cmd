@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Public launcher. Keep startup logic in START_PROGRAM_INTERNAL.cmd.
if not exist "%~dp0START_PROGRAM_INTERNAL.cmd" (
    echo.
    echo ERROR: START_PROGRAM_INTERNAL.cmd was not found.
    echo Please extract the complete ZIP archive before running.
    echo.
    pause
    exit /b 20
)

call "%~dp0START_PROGRAM_INTERNAL.cmd"
exit /b %ERRORLEVEL%
