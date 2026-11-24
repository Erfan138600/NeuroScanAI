@echo off
chcp 65001 >nul
echo ==================================================================
echo 🧠 NeuroScan AI Pro v3.2 - WindowsPyInstaller EXE Builder
echo 👨‍💻 برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
echo ==================================================================

pip install pyinstaller opencv-python torch torchvision numpy pillow

pyinstaller --noconfirm --onefile --name "NeuroScanAI_Server_ErfanAlikhani" ^
    --add-data "web;web" ^
    --add-data "samples;samples" ^
    --add-data "core;core" ^
    server.py

echo.
echo ✅ فایل اجرایی درون پوشه dist ساخته شد!
pause
