# ติดตั้งฟอนต์ไทยที่มากับโปรแกรม (app/assets/fonts) เข้า Windows
#
# โปรแกรมนับยาเองไม่ต้องใช้สคริปต์นี้ -- มันโหลดไฟล์ฟอนต์จากโฟลเดอร์ของตัวเองตอนเปิด
# (theme.load_fonts) ไม่ต้องต่อเน็ตและไม่ต้องติดตั้งอะไร สคริปต์นี้มีไว้เพื่อให้
# โปรแกรมอื่นในเครื่อง -- Word, Excel, เบราว์เซอร์ -- เห็นฟอนต์ตัวเดียวกันด้วย
#
# รันแบบ Administrator = ติดตั้งให้ทุกผู้ใช้ (C:\Windows\Fonts)
# รันแบบผู้ใช้ธรรมดา   = ติดตั้งเฉพาะผู้ใช้ปัจจุบัน (%LOCALAPPDATA%) ไม่ต้องขอสิทธิ์
#
# ถอนการติดตั้ง: ลบไฟล์ในโฟลเดอร์ฟอนต์ แล้วลบค่าที่ชื่อขึ้นต้นด้วย "IBM Plex Sans Thai"
# ออกจากคีย์รีจิสทรีที่สคริปต์นี้พิมพ์ออกมา

$ErrorActionPreference = "Stop"

$source = Join-Path $PSScriptRoot "app\assets\fonts"
$files = Get-ChildItem -Path (Join-Path $source "*.ttf") -ErrorAction SilentlyContinue
if (-not $files) {
    Write-Host "ไม่พบไฟล์ฟอนต์ใน $source"
    exit 1
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if ($admin) {
    $target = Join-Path $env:WINDIR "Fonts"
    $key = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
    Write-Host "ติดตั้งให้ทุกผู้ใช้:" $target
} else {
    $target = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Fonts"
    $key = "HKCU:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
    Write-Host "ติดตั้งเฉพาะผู้ใช้นี้:" $target
}

New-Item -ItemType Directory -Force -Path $target | Out-Null
if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }

foreach ($file in $files) {
    $installed = Join-Path $target $file.Name
    Copy-Item -Path $file.FullName -Destination $installed -Force

    # ชื่อที่ Windows ใช้แสดงในรายการฟอนต์ เช่น "IBM Plex Sans Thai SemiBold (TrueType)"
    # ชื่อหน้าตัวอักษรมาจากชื่อไฟล์ตรง ๆ และห้ามแทรกช่องว่างเข้าไปกลางคำ:
    # Windows รู้จัก "SemiBold" ไม่ใช่ "Semi Bold" และถ้าชื่อไม่ตรง ฟอนต์จะโผล่เป็น
    # แฟมิลีแยกในบางโปรแกรม
    $face = [IO.Path]::GetFileNameWithoutExtension($file.Name) -replace "^IBMPlexSansThai-?", ""
    $display = ("IBM Plex Sans Thai " + $face).Trim() + " (TrueType)"

    # รีจิสทรีของเครื่องเก็บแค่ชื่อไฟล์ (มันอยู่ใน C:\Windows\Fonts อยู่แล้ว)
    # ส่วนของผู้ใช้ต้องเก็บ path เต็ม ไม่งั้น Windows หาไฟล์ไม่เจอ
    $value = if ($admin) { $file.Name } else { $installed }
    New-ItemProperty -Path $key -Name $display -Value $value `
        -PropertyType String -Force | Out-Null
    Write-Host "  ติดตั้งแล้ว: $display"
}

Write-Host ""
Write-Host "เสร็จแล้ว  คีย์รีจิสทรี: $key"
Write-Host "โปรแกรมที่เปิดค้างอยู่ต้องปิดแล้วเปิดใหม่จึงจะเห็นฟอนต์"
