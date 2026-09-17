@echo off
REM หน้าจอใช้งานจริง -- กล้องขวา ตัวเลขซ้าย ตั้งเป้าหมาย บันทึก JSON
cd /d "%~dp0"
"D:\pill-counter-lite\pillcount-v12\.venv-train\Scripts\python.exe" -m app %*
