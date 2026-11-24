@echo OFF
REM ===========================================================================
REM NeuroScan AI Ultra v4.0 - Standalone Windows x64 Build Script
REM برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
REM ===========================================================================
REM Requires: MinGW-w64 g++ (x86_64) in PATH  e.g.  C:\mingw64\bin
REM Output  : NeuroScanAI_Ultra_v4_Erfan_Alikhani.exe  (100% standalone)
REM ===========================================================================
setlocal
set SRC=neuroscan_win32.cpp
set OUT=NeuroScanAI_Ultra_v4_Erfan_Alikhani.exe

echo [1/3] Compiling Windows resource (icon / version info)...
windres app_info.rc -O coff -o app_info.res 2>nul

echo [2/3] Compiling C++17 native engine + Win32 GUI (static, zero DLLs)...
x86_64-w64-mingw32-g++ -O3 -std=c++17 -municode -mwindows ^
  -o %OUT% %SRC% app_info.res ^
  -lgdiplus -lshlwapi -lcomdlg32 -lshell32 -lole32 ^
  -static -static-libgcc -static-libstdc++

if errorlevel 1 (
  echo BUILD FAILED!
  exit /b 1
)

echo [3/3] Verifying DLL dependency list (only Windows system DLLs expected)...
x86_64-w64-mingw32-objdump -p %OUT% | findstr "DLL Name"

echo.
echo ============================================================
echo  BUILD OK:  %OUT%
echo  Developer: Erfan Alikhani (برنامه نویس عرفان علیخانی)
echo ============================================================
endlocal
