@echo off
REM สร้างไฟล์ APK ของ model3 (โมเดล v3 ตัวเดียวกับที่ใช้บนพีซี รันบนเครื่องโทรศัพท์เอง)
REM ผลลัพธ์อยู่ที่ model3\android\dist\DrugCount.apk
REM
REM หน้าจอบนมือถือเขียนอยู่ใน model3\phone\ ดูหน้าตาบนพีซีได้โดยไม่ต้อง build:
REM     python -m model3.tools.preview
REM
REM ต้องมี JDK 17 และ Android SDK  ถ้าขาดอะไร build.ps1 จะบอกว่าต้องติดตั้งอะไร
REM
REM   4-สร้างไฟล์APK.bat              สร้างอย่างเดียว
REM   4-สร้างไฟล์APK.bat -Install     สร้างแล้วติดตั้งลงมือถือที่เสียบ USB อยู่
REM   4-สร้างไฟล์APK.bat -Release     สร้างแบบ release (ยังเซ็นด้วย debug key)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0model3\android\build.ps1" %*
pause
