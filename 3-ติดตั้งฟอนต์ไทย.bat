@echo off
REM ติดตั้งฟอนต์ไทยที่มากับโปรแกรม เข้า Windows เพื่อให้โปรแกรมอื่นใช้ได้ด้วย
REM โปรแกรมนับยาไม่ต้องใช้ไฟล์นี้ -- มันอ่านฟอนต์จาก app\assets\fonts เองตอนเปิด
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-fonts.ps1"
pause
