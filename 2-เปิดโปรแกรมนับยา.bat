@echo off
REM หน้าจอใช้งานจริง -- กล้องขวา ตัวเลขซ้าย ตั้งเป้าหมาย บันทึก JSON
cd /d "%~dp0"
call "%~dp0find-python.bat" || exit /b 1
"%PY%" -m app %*
if errorlevel 1 pause
