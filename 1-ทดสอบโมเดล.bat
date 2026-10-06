@echo off
REM เครื่องมือวัด -- เปิดกล้อง ไม่มี UI ไม่มีปุ่ม มีแต่ตัวเลข
REM ใช้ตอนอยากรู้ว่าโมเดลแม่นและนิ่งแค่ไหน ตอนปิดจะพิมพ์ spread ออกมา
REM   q / Esc  ออก      r  วาดขอบเขตใหม่      s  เก็บภาพ + พิกัดกล่อง
cd /d "%~dp0"
call "%~dp0find-python.bat" || exit /b 1
"%PY%" model3\bench.py %*
echo.
pause
