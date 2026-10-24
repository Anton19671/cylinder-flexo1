@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel%==0 (
    python tests\LAYOUT_STRUCTURE_SELF_CHECK.py
) else (
    py -3 tests\LAYOUT_STRUCTURE_SELF_CHECK.py
)
pause
