@echo off
cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel%==0 (
  python FINAL_SELF_CHECK.py
) else (
  py -3 FINAL_SELF_CHECK.py
)
pause
