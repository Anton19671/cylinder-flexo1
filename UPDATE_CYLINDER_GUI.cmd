@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "CYL_GUI_ROOT=%~dp0"
rem This updates only interface/FINAL_GUI.py in the running program folder.
rem The solver, physical CASE files, and previous results are not changed.
if not exist "%CYL_GUI_ROOT%FINAL_GUI.py" goto :wrongfolder
if not exist "%CYL_GUI_ROOT%interface\FINAL_GUI.py" goto :wrongfolder
set "CYL_GUI_SOURCE=https://raw.githubusercontent.com/Anton19671/cylinder-flexo1/cc7d3871de079f29a3abace42eb6a1e8aeaeff7a/interface/FINAL_GUI.py"
echo.
echo Cylinder-Flexo1: restore the complete 8-tab GUI.
echo Please close the running GUI before continuing.
pause
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $dest=Join-Path $env:CYL_GUI_ROOT 'interface\FINAL_GUI.py'; $temp=$dest+'.download'; try { Invoke-WebRequest -Uri $env:CYL_GUI_SOURCE -OutFile $temp -UseBasicParsing; $text=[IO.File]::ReadAllText($temp,[Text.Encoding]::UTF8); if (!$text.Contains('class ElasticFlexoGUI') -or !$text.Contains('self.tab_online=self._tab')) { throw 'Wrong downloaded file' }; $backup=$dest+'.before_gui_restore_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.bak'; Copy-Item -LiteralPath $dest -Destination $backup -ErrorAction Stop; Move-Item -LiteralPath $temp -Destination $dest -Force; Write-Host 'UPDATED:' $dest; Write-Host 'BACKUP:' $backup } catch { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue; Write-Host 'UPDATE FAILED:' $_.Exception.Message -ForegroundColor Red; exit 1 }"
if errorlevel 1 goto :failed
echo.
echo GUI updated. Start START_PROGRAM.bat from this same folder.
echo The title must contain: G3D / Gz - full interface.
pause
exit /b 0
:wrongfolder
echo.
echo WRONG FOLDER: put this CMD file alongside START_PROGRAM.bat.
echo No files were changed.
pause
exit /b 2
:failed
echo.
echo Update failed. Your old GUI remains available as a backup.
pause
exit /b 1
