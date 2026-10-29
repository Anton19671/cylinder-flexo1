@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" || exit /b 2
set "SCRIPT_ROOT=%CD%"
rem V-11: NEVER replace interface/FINAL_GUI.py from an old pinned revision.
rem GUI, solver, config and tests must come from the SAME repository commit.
where git.exe >nul 2>&1
if errorlevel 1 goto :no_git
for /f "delims=" %%R in ('git rev-parse --show-toplevel 2^>nul') do set "GIT_ROOT=%%R"
if not defined GIT_ROOT goto :not_checkout
pushd "%GIT_ROOT%" || exit /b 2
if /I not "%SCRIPT_ROOT%"=="%CD%" goto :wrong_root
if not exist "interface\FINAL_GUI.py" goto :wrong_root
if not exist "interface\cylinder_reference_case.py" goto :wrong_root
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if /I not "%CURRENT_BRANCH%"=="main" goto :not_main
git remote get-url origin 2>nul | findstr /i /c:"github.com/Anton19671/cylinder-flexo1" >nul
if errorlevel 1 goto :wrong_remote
git diff --quiet --exit-code
if errorlevel 1 goto :dirty
git diff --cached --quiet --exit-code
if errorlevel 1 goto :dirty
echo.
echo Cylinder-Flexo1: updating the COMPLETE repository, including GUI and solver.
echo Close the running program before continuing.
echo CASE inputs and previous results are not deleted.
git pull --ff-only origin main
if errorlevel 1 goto :pull_failed
echo.
echo Repository updated. Launch START_PROGRAM.bat again.
popd
exit /b 0

:not_checkout
echo ERROR: not a Git checkout. Single-file GUI update has been disabled.
echo Download the COMPLETE current repository from:
echo https://github.com/Anton19671/cylinder-flexo1
exit /b 2
:no_git
echo ERROR: git.exe unavailable. Download the COMPLETE repository archive.
exit /b 2
:not_main
echo ERROR: expected branch main, found %CURRENT_BRANCH%. No files changed.
goto :fail_pop
:wrong_remote
echo ERROR: origin is not Anton19671/cylinder-flexo1. No files changed.
goto :fail_pop
:dirty
echo ERROR: local tracked changes exist. Commit or back them up first.
goto :fail_pop
:wrong_root
echo ERROR: expected the cylinder-flexo1 repository root.
goto :fail_pop
:pull_failed
echo ERROR: git pull failed; no forced replacement attempted.
:fail_pop
popd
exit /b 2
