@echo off
REM ดูข้อความจากแอปบนมือถือที่เสียบสาย USB อยู่ (ต้องเปิด USB debugging)
REM
REM ใช้ตอนแอปบนมือถือมีอาการผิดปกติ เช่นขึ้นว่า "โมเดลผิดพลาด" -- traceback เต็ม ๆ
REM จะอยู่ในนี้ ไม่ใช่บนหน้าจอ
REM
REM ปิดด้วย Ctrl+C
cd /d "%~dp0"
set ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe
"%ADB%" logcat -c
echo กำลังรอข้อความจากแอป... เปิดแอปบนมือถือได้เลย
"%ADB%" logcat -s python.stdout:* python.stderr:* PILLSORT:* AndroidRuntime:E
