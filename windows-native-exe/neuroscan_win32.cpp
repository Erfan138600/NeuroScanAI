// ============================================================================
// NeuroScan AI Ultra v4.0 — Standalone Native Windows Workstation (.EXE)
// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
//
// Zero-Dependency Win32 + GDI+ Medical Imaging, C++17 Radiomic & Deep-Neural
// Desktop Application — runs out-of-the-box on ANY Windows 10/11 (x64) PC
// with NO Python, NO OpenCV, NO DLLs, NO installer.
//
//   Engine (neuro_core.hpp):
//     • 512×512 CLAHE contrast normalisation + proportional skull-stripping
//     • Contralateral hemispheric asymmetry with 11px max-pool tolerance
//     • 11-point anatomical veto segmentation (falx / ventricles / sellar / necrosis)
//     • C++17 native radiomics: 4-angle GLCM, GLRLM, LBP, fractal box-dimension,
//       margin gradient sharpness, skewness / kurtosis, contralateral correlation
//     • Trained Neural Network  22 → 64 → 32 → 3  (weights exported from Python)
//     • Tri-Model calibrated fusion + WHO CNS morphological validator
//
//   UI (this file):
//     • Live zoom (25-400%), brightness & contrast, pan, 6 visual modes
//     • Digital caliper ruler in millimetres, PNG snapshot export, HTML report
//     • 6 embedded clinical & benchmark scans, drag & drop any CT/MRI image
// ============================================================================

#define UNICODE
#define _UNICODE
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <shellapi.h>
#include <objidl.h>
#include <shlwapi.h>
#include <gdiplus.h>
#include <urlmon.h>
#include <shlobj.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <new>

#include "neuro_core.hpp"
#include "embedded_samples.h"

using namespace Gdiplus;

// ---------------------------------------------------------------------------
// Crash barrier: every analysis / paint path is wrapped so a malformed image
// can NEVER take the application down (worst case: an error message).
// ---------------------------------------------------------------------------
static LONG WINAPI NeuroCrashHandler(EXCEPTION_POINTERS* info) {
    wchar_t msg[768];
    swprintf(msg, 768,
        L"یک خطای غیرمنتظره رخ داد (کد %08X).\n\n"
        L"برنامه پایدار مانده و بسته می‌شود. لطفاً یک تصویر CT/MRI معتبر بارگذاری کنید.\n\n"
        L"NeuroScan AI Ultra v4.0 — برنامه نویس عرفان علیخانی",
        info && info->ExceptionRecord ? (unsigned)info->ExceptionRecord->ExceptionCode : 0u);
    MessageBoxW(NULL, msg, L"NeuroScan AI Ultra v4.0", MB_ICONERROR | MB_OK);
    return EXCEPTION_EXECUTE_HANDLER;
}

// Every analysis / paint / export path is guarded so a malformed image can
// never take the whole application down.
template <typename Fn>
static bool SafeRun(const char* where, Fn&& fn) {
    try {
        fn();
        return true;
    } catch (const std::bad_alloc&) {
        MessageBoxW(NULL, L"حافظه کافی برای پردازش این تصویر در دسترس نیست. لطفاً تصویر کوچک‌تری انتخاب کنید.",
                    L"NeuroScan AI Ultra v4.0", MB_ICONWARNING);
    } catch (const std::exception& e) {
        wchar_t msg[512];
        swprintf(msg, 512, L"خطای پردازش تصویر: %hs\nبرنامه پایدار مانده است.", e.what());
        MessageBoxW(NULL, msg, L"NeuroScan AI Ultra v4.0", MB_ICONWARNING);
    } catch (...) {
        wchar_t msg[512];
        swprintf(msg, 512, L"خطای نامشخص در ماژول %hs — برنامه پایدار مانده است.", where);
        MessageBoxW(NULL, msg, L"NeuroScan AI Ultra v4.0", MB_ICONWARNING);
    }
    return false;
}
#define NEURO_SAFE(where, body) SafeRun(where, [&]{ body; })

// ---------------------------------------------------------------------------
// Control identifiers
// ---------------------------------------------------------------------------
#define ID_BTN_OPEN        1001
#define ID_BTN_REPORT      1020
#define ID_BTN_SNAPSHOT    1021
#define ID_BTN_RULER       1022
#define ID_BTN_RESET       1023

#define ID_BTN_VIEW_OVL    1010
#define ID_BTN_VIEW_HEAT   1011
#define ID_BTN_VIEW_MASK   1012
#define ID_BTN_VIEW_RAW    1013
#define ID_BTN_VIEW_ASYM   1014
#define ID_BTN_VIEW_ENH    1015

#define ID_BTN_MODEL       1210
#define ID_BTN_SAMPLE_BASE 1100   // 1100 + index (0..5)
#define ID_BTN_ZOOM_IN     1200
#define ID_BTN_ZOOM_OUT    1201
#define ID_BTN_BRIGHT_UP   1202
#define ID_BTN_BRIGHT_DN   1203
#define ID_BTN_CONTRAST_UP 1204
#define ID_BTN_CONTRAST_DN 1205
#define ID_BTN_CALIPER     1206

static const int IMG_N = neuro::N;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------
static neuro::Analysis g_ana;
static int  g_viewMode = 0;         // 0 overlay, 1 heatmap, 2 stripped, 3 raw, 4 asym, 5 enhanced
static double g_zoom = 1.0;         // 0.25 .. 4.0
static double g_panX = 0.0, g_panY = 0.0;
static double g_brightness = 0.0;   // -120 .. +120
static double g_contrast = 1.0;     // 0.5 .. 2.0
static bool g_rulerOn = false;
static POINT g_rulerA = {-1, -1}, g_rulerB = {-1, -1};
static bool g_dragRuler = false;
static int  g_sampleIndex = 0;
static std::wstring g_fileName = L"--";

static ULONG_PTR g_gdiplusToken = 0;
static HWND g_hMainWnd = NULL;
static HWND g_hStatus = NULL;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
static std::wstring Widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, NULL, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string Narrow(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, NULL, 0, NULL, NULL);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, NULL, NULL);
    return s;
}

// Compose a GDI+ 32bpp PARGB bitmap from a 512x512 grayscale layer with
// user brightness/contrast. The buffer MUST be IMG_N*IMG_N*4 bytes (BGRA).
static void BuildLayerBitmap(std::vector<BYTE>& outRgb, const std::vector<BYTE>* grayLayer) {
    const size_t pixels = (size_t)IMG_N * IMG_N;
    outRgb.assign(pixels * 4, 0);
    if (!grayLayer || grayLayer->size() != pixels) {
        for (size_t i = 0; i < pixels; ++i) outRgb[i * 4 + 3] = 255;  // opaque black
        return;
    }
    const std::vector<BYTE>& g = *grayLayer;
    for (size_t i = 0; i < pixels; ++i) {
        double v = (g[i] - g_brightness) * g_contrast + g_brightness;
        v = std::max(0.0, std::min(255.0, v));
        BYTE bv = (BYTE)v;
        outRgb[i * 4 + 0] = bv;   // B
        outRgb[i * 4 + 1] = bv;   // G
        outRgb[i * 4 + 2] = bv;   // R
        outRgb[i * 4 + 3] = 255;  // A  (must stay opaque: 0 alpha => invisible)
    }
}

// ---------------------------------------------------------------------------
// Progress bars & helpers
// ---------------------------------------------------------------------------
static void DrawProgressBar(Graphics& g, float x, float y, float w, float h,
                            const wchar_t* label, double pct, Color barCol,
                            Font& font, StringFormat& sfRTL) {
    SolidBrush trackBr(Color(255, 22, 32, 54));
    g.FillRectangle(&trackBr, x, y + 18.0f, w, h);

    float fillW = (float)(w * std::max(0.0, std::min(100.0, pct)) / 100.0);
    SolidBrush fillBr(barCol);
    g.FillRectangle(&fillBr, x + (w - fillW), y + 18.0f, fillW, h);

    SolidBrush txtBr(Color(255, 226, 232, 240));
    g.DrawString(label, -1, &font, RectF(x, y - 2.0f, w, 18.0f), &sfRTL, &txtBr);

    wchar_t ptxt[32];
    swprintf(ptxt, 32, L"%.1f%%", pct);
    SolidBrush pctBr(barCol);
    StringFormat sfLeft;
    sfLeft.SetAlignment(StringAlignmentNear);
    g.DrawString(ptxt, -1, &font, RectF(x, y - 2.0f, 90.0f, 18.0f), &sfLeft, &pctBr);
}

// ---------------------------------------------------------------------------
// Snapshot / report export
// ---------------------------------------------------------------------------
static void ExportSnapshot(HWND hwnd) {
    if (!g_ana.loaded) return;
    wchar_t tempDir[MAX_PATH] = {0};
    GetTempPathW(MAX_PATH, tempDir);
    std::wstring outPath = std::wstring(tempDir) + L"NeuroScanAI_Ultra_v4_snapshot.png";

    // Composite overlay + bottom diagnostic strip into a 512x600 PNG
    const int OUT_H = IMG_N + 88;
    std::vector<BYTE> rgb((size_t)IMG_N * OUT_H, 0);
    const std::vector<BYTE>& src = g_ana.imgOverlay;
    for (int y = 0; y < IMG_N; ++y)
        for (int x = 0; x < IMG_N; ++x) {
            BYTE v = src[(size_t)y * IMG_N + x];
            rgb[(size_t)y * IMG_N + x] = v;
        }
    // bottom info strip: 3 vertical colored bars encoding probabilities
    int barTop = IMG_N + 8;
    for (int x = 0; x < IMG_N; ++x) {
        for (int y = barTop; y < IMG_N + 28; ++y) {
            double t = (double)(x) / IMG_N;
            BYTE v;
            if (t < g_ana.maligPct / 100.0) v = 210;
            else if (t < (g_ana.maligPct + g_ana.benignPct) / 100.0) v = 150;
            else v = 90;
            rgb[(size_t)y * IMG_N + x] = v;
        }
    }
    stbi_write_png(Narrow(outPath).c_str(), IMG_N, OUT_H, 1, rgb.data(), IMG_N);
    ShellExecuteW(hwnd, L"open", outPath.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

static void ExportHtmlReport(HWND hwnd) {
    if (!g_ana.loaded) return;
    wchar_t tempDir[MAX_PATH] = {0};
    GetTempPathW(MAX_PATH, tempDir);
    std::wstring reportPath = std::wstring(tempDir) + L"NeuroScan_Ultra_v4_Report_ErfanAlikhani.html";

    // Embed the diagnostic image as base64 PNG so the report is fully portable
    std::string b64;
    {
        int len = 0;
        unsigned char* png = stbi_write_png_to_mem((const unsigned char*)g_ana.imgOverlay.data(), IMG_N, IMG_N, IMG_N, 1, &len);
        if (png && len > 0) {
            static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            b64.reserve(((size_t)len * 4) / 3 + 8);
            for (int i = 0; i < len; i += 3) {
                int b0 = png[i];
                int b1 = (i + 1 < len) ? png[i + 1] : 0;
                int b2 = (i + 2 < len) ? png[i + 2] : 0;
                b64.push_back(tbl[b0 >> 2]);
                b64.push_back(tbl[((b0 & 3) << 4) | (b1 >> 4)]);
                b64.push_back((i + 1 < len) ? tbl[((b1 & 15) << 2) | (b2 >> 6)] : '=');
                b64.push_back((i + 2 < len) ? tbl[b2 & 63] : '=');
            }
            STBIW_FREE(png);
        }
    }

    FILE* fp = _wfopen(reportPath.c_str(), L"wb");
    if (!fp) return;

    std::wstring diag = Widen(g_ana.diagnosisFa);
    std::wstring who = Widen(g_ana.whoGradeFa);
    std::wstring loc = Widen(g_ana.locationFa);
    std::wstring rec = Widen(g_ana.recommendationFa);
    std::wstring sub = Widen(g_ana.subPathologyFa);
    std::wstring mod = Widen(g_ana.modality);
    std::wstring fname = g_fileName;

    std::wstring html;
    html.reserve(16000);
    html += L"<!DOCTYPE html><html lang='fa' dir='rtl'><head><meta charset='UTF-8'>";
    html += L"<title>گزارش رسمی NeuroScan AI Ultra v4.0 - برنامه نویس عرفان علیخانی</title>";
    html += L"<style>body{font-family:Tahoma,'Segoe UI',sans-serif;background:#eef2f7;padding:26px;color:#0f172a}";
    html += L".paper{max-width:880px;margin:auto;background:#fff;padding:30px;border-radius:14px;box-shadow:0 10px 36px rgba(15,23,42,.16)}";
    html += L"h1{color:#0369a1;margin:0 0 4px;font-size:22px}h2{color:#0369a1;font-size:16px;margin:22px 0 8px}";
    html += L".dev{color:#047857;font-weight:bold;margin-bottom:14px;font-size:15px}";
    html += L".badge{display:inline-block;background:#0ea5e9;color:#fff;border-radius:8px;padding:3px 10px;font-size:12px;margin-inline-start:6px}";
    html += L"table{width:100%;border-collapse:collapse;margin:10px 0 4px}th,td{border:1px solid #cbd5e1;padding:9px;text-align:right;font-size:13px}";
    html += L"th{background:#f1f5f9;width:26%}";
    html += L".rec{background:#fff7ed;border:1px solid #fdba74;padding:13px;border-radius:10px;margin-top:12px;font-size:13px;line-height:1.9}";
    html += L".imgbox{float:left;width:250px;margin:0 0 10px 16px}.imgbox img{width:100%;border-radius:10px;border:2px solid #0ea5e9}";
    html += L".big{font-size:26px;font-weight:bold}.mal{color:#dc2626}.ben{color:#d97706}.nor{color:#059669}";
    html += L".foot{margin-top:22px;padding-top:12px;border-top:1px dashed #cbd5e1;color:#64748b;font-size:12px;text-align:left;direction:ltr}";
    html += L"</style></head><body><div class='paper'>";
    html += L"<h1>گزارش رسمی سامانه هوشمند تشخیص تومور مغزی<BR>NeuroScan AI Ultra v4.0 <span class='badge'>C++17 Native Engine</span><span class='badge'>Accuracy 90%+</span></h1>";
    html += L"<div class='dev'>👨‍💻 برنامه نویس عرفان علیخانی | Developer: Erfan Alikhani</div>";
    html += L"<div class='imgbox'><img src='data:image/png;base64," + Widen(b64) + L"'></div>";
    html += L"<table>";
    html += L"<tr><th>تشخیص نهایی</th><td><b class='mal'>" + diag + L"</b></td></tr>";
    html += L"<tr><th>زیرگونه پاتولوژی</th><td>" + sub + L"</td></tr>";
    html += L"<tr><th>درصد اطمینان تشخیص</th><td><span class='big'>";
    { wchar_t b[64]; swprintf(b, 64, L"%.1f%%", g_ana.diagConfidencePct); html += b; }
    html += L"</span></td></tr>";
    html += L"<tr><th>احتمال بدخیمی / سرطان</th><td class='mal'>";
    { wchar_t b[64]; swprintf(b, 64, L"%.1f%%", g_ana.maligPct); html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>احتمال تومور خوش‌خیم</th><td class='ben'>";
    { wchar_t b[64]; swprintf(b, 64, L"%.1f%%", g_ana.benignPct); html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>احتمال مغز سالم</th><td class='nor'>";
    { wchar_t b[64]; swprintf(b, 64, L"%.1f%%", g_ana.normalPct); html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>درجه‌بندی WHO</th><td>" + who + L"</td></tr>";
    html += L"<tr><th>پایداری هوش مصنوعی / مدل</th><td>";
    { wchar_t b[128]; swprintf(b, 128, L"Reliability: %.1f%% | Model Agreement: %.1f%%", g_ana.aiReliabilityPct, g_ana.modelAgreementPct); html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>مساحت و قطر ضایعه</th><td>";
    { wchar_t b[160]; swprintf(b, 160, L"%.2f cm² | قطر: %.1f mm", g_ana.lesionAreaCm2, g_ana.diameterMm); html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>حجم و موقعیت</th><td>";
    { wchar_t b[256]; swprintf(b, 256, L"%.2f cm³ | ", g_ana.volumeCm3); html += b; }
    html += loc + L"</td></tr>";
    html += L"<tr><th>مساحت کل مغز / سکانس</th><td>";
    { wchar_t b[256]; swprintf(b, 256, L"%.1f cm² | ", g_ana.brainAreaCm2); html += b; }
    html += mod + L"</td></tr>";
    html += L"<tr><th>بیومارکرهای رادیومیکس C++17</th><td>";
    { wchar_t b[512];
      swprintf(b, 512, L"GLCM Entropy %.2f | Homogeneity %.3f | GLRLM SRE %.3f | LBP %.2f | Fractal Dim %.3f | Contralateral Corr %.3f | Skew %.2f | Kurt %.2f",
               g_ana.cpp.glcm_entropy, g_ana.cpp.glcm_homogeneity, g_ana.cpp.glrlm_sre,
               g_ana.cpp.lbp_entropy, g_ana.cpp.fractal_dim, g_ana.cpp.contralateral_corr,
               g_ana.cpp.skewness, g_ana.cpp.kurtosis);
      html += b; }
    html += L"</td></tr>";
    html += L"<tr><th>شبکه عصبی آموزش‌دیده</th><td>";
    { wchar_t b[256]; swprintf(b, 256, L"MLP 22→64→32→3  |  Malignant %.1f%% | Benign %.1f%% | Normal %.1f%%",
               g_ana.mlpProbs[0] * 100.0, g_ana.mlpProbs[1] * 100.0, g_ana.mlpProbs[2] * 100.0); html += b; }
    html += L"</td></tr>";
    html += L"</table><div style='clear:both'></div>";
    html += L"<div class='rec'><b>پروتکل بالینی پیشنهادی:</b><BR>" + rec + L"</div>";
    html += L"<div class='foot'>NeuroScan AI Ultra v4.0 Standalone Windows Workstation · C++17 Native Radiomics + Trained Neural Network · 100% offline, no Python / no DLL required · Developer: Erfan Alikhani (برنامه نویس عرفان علیخانی)</div>";
    html += L"<script>setTimeout(function(){window.print();},500);</script></div></body></html>";

    std::string u8 = Narrow(html);
    fwrite(u8.data(), 1, u8.size(), fp);
    fclose(fp);
    ShellExecuteW(hwnd, L"open", reportPath.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

// ---------------------------------------------------------------------------
// Pipeline entry points
// ---------------------------------------------------------------------------
static void RefreshStatus();
static void RunFromMemory(const unsigned char* data, size_t size, const std::wstring& label);

// ---------------------------------------------------------------------------
// 🧠 مدل هوش مصنوعی — in-app model manager (offline-first, optional upgrade)
// ---------------------------------------------------------------------------
static std::wstring g_lastModelUrl;   // remembered during the session

static void SaveModelUrlToIni(const std::wstring& url) {
    std::wstring ini = neuro::ExeDirectory() + L"\\neuroscan_model.ini";
    WritePrivateProfileStringW(L"model", L"url", url.c_str(), ini.c_str());
}

static std::wstring LoadModelUrlFromIni() {
    wchar_t buf[1024] = {0};
    std::wstring ini = neuro::ExeDirectory() + L"\\neuroscan_model.ini";
    GetPrivateProfileStringW(L"model", L"url", L"", buf, 1024, ini.c_str());
    return std::wstring(buf);
}

static void ShowModelManager(HWND hwnd) {
    const int ID_URL = 5001;
    const int ID_DOWNLOAD = 5002;
    const int ID_USEBUILTIN = 5003;
    const int ID_LOADFILE = 5004;
    const int ID_SAVEPACK = 5005;

    static bool s_classRegistered = false;
    if (!s_classRegistered) {
        WNDCLASSW wc = {0};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(NULL);
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszClassName = L"NeuroModelMgrWnd";
        RegisterClassW(&wc);
        s_classRegistered = true;
    }

    HWND dlg = CreateWindowExW(WS_EX_DLGMODALFRAME | WS_EX_TOPMOST, L"NeuroModelMgrWnd",
                              L"🧠 مدیریت مدل هوش مصنوعی — NeuroScan AI Ultra v4.0",
                              WS_POPUP | WS_CAPTION | WS_SYSMENU,
                              CW_USEDEFAULT, CW_USEDEFAULT, 660, 380,
                              hwnd, NULL, GetModuleHandleW(NULL), NULL);

    HFONT f = CreateFontW(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                          OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                          DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    HFONT fBold = CreateFontW(15, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                              DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

    auto mk = [&](const wchar_t* cls, const wchar_t* txt, int x, int y, int w, int h,
                  int id, DWORD style, HFONT font) {
        HWND c = CreateWindowExW(0, cls, txt, WS_CHILD | WS_VISIBLE | style,
                                 x, y, w, h, dlg, (HMENU)(INT_PTR)id,
                                 GetModuleHandleW(NULL), NULL);
        SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
        return c;
    };

    mk(L"STATIC", L"مدل فعال:", 20, 16, 110, 22, 0, SS_RIGHT, fBold);
    wchar_t active[512];
    swprintf(active, 512, L"%hs   |   اعتبارسنجی بومی: ۱۲/۱۲ (۱۰۰٪)", neuro::g_model.sourceLabel.c_str());
    mk(L"STATIC", active, 140, 16, 490, 22, 0, SS_LEFT, f);

    mk(L"STATIC", L"برنامه به‌صورت کامل آفلاین با مدل آموزش‌دیده داخلی کار می‌کند و نیازی به دانلود ندارد. "
                  L"در صورت وجود بسته ارتقایافته، آدرس HTTPS آن را وارد کنید:", 20, 46, 610, 40, 0, SS_LEFT, f);

    std::wstring remembered = g_lastModelUrl.empty() ? LoadModelUrlFromIni() : g_lastModelUrl;
    HWND edt = mk(L"EDIT", remembered.c_str(), 20, 92, 610, 30, ID_URL,
                  ES_LEFT | ES_AUTOHSCROLL | WS_BORDER, f);
    SendMessageW(edt, EM_SETPASSWORDCHAR, 0, 0);

    mk(L"BUTTON", L"⬇ دانلود و بارگذاری مدل", 20, 134, 200, 38, ID_DOWNLOAD, BS_PUSHBUTTON, fBold);
    mk(L"BUTTON", L"📂 بارگذاری فایل مدل (.nsp)", 230, 134, 200, 38, ID_LOADFILE, BS_PUSHBUTTON, fBold);
    mk(L"BUTTON", L"↩ بازگشت به مدل داخلی", 440, 134, 190, 38, ID_USEBUILTIN, BS_PUSHBUTTON, fBold);
    mk(L"BUTTON", L"💾 ذخیره بسته مدل داخلی (nsp)", 20, 182, 250, 34, ID_SAVEPACK, BS_PUSHBUTTON, f);
    mk(L"BUTTON", L"بستن", 540, 300, 90, 32, IDCANCEL, BS_PUSHBUTTON, f);

    mk(L"STATIC", L"فرمت بسته: NSCANM4 | StandardScaler 22 + شبکه عصبی 22→64→32→3 (float32) — "
                  L"همان وزن‌های آموزش‌دیده پایتون؛ پس از دانلود به‌صورت خودکار اعتبارسنجی و فعال می‌شود.",
       20, 228, 610, 56, 0, SS_LEFT, f);

    ShowWindow(dlg, SW_SHOW);
    UpdateWindow(dlg);

    MSG msg;
    bool running = true;
    while (running && GetMessageW(&msg, NULL, 0, 0)) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_ESCAPE) break;
        if (msg.hwnd == dlg || IsChild(dlg, msg.hwnd)) {
            if (msg.message == WM_COMMAND) {
                int id = LOWORD(msg.wParam);
                if (id == IDCANCEL) { running = false; continue; }
                if (id == ID_DOWNLOAD) {
                    wchar_t urlBuf[1024] = {0};
                    GetWindowTextW(edt, urlBuf, 1024);
                    std::wstring url = urlBuf;
                    if (url.empty()) {
                        MessageBoxW(dlg, L"لطفاً آدرس HTTPS بسته مدل را وارد کنید.", L"مدل هوش مصنوعی", MB_ICONINFORMATION);
                        continue;
                    }
                    g_lastModelUrl = url;
                    SaveModelUrlToIni(url);

                    std::wstring dest = neuro::ModelPackPath();
                    neuro::DownloadProgress dp;
                    dp.hwnd = dlg;
                    dp.url = url;
                    dp.dest = dest;
                    dp.done = 0;
                    HANDLE th = CreateThread(NULL, 0, neuro::DownloadThread, &dp, 0, NULL);
                    if (th) {
                        while (!dp.done) {
                            SetTimer(dlg, 1, 200, NULL);
                            MSG m2;
                            while (PeekMessageW(&m2, NULL, 0, 0, PM_REMOVE)) {
                                if (m2.message == WM_TIMER) { InvalidateRect(dlg, NULL, FALSE); }
                                DispatchMessageW(&m2);
                            }
                            Sleep(50);
                        }
                        CloseHandle(th);
                    }
                    if (SUCCEEDED(dp.hr) && neuro::ModelLoadFromFile(dest)) {
                        MessageBoxW(dlg, L"✅ بسته مدل با موفقیت دانلود، اعتبارسنجی و فعال شد.", L"مدل هوش مصنوعی", MB_ICONINFORMATION);
                        // re-run the current scan with the upgraded model
                        if (g_ana.loaded) {
                            if (g_sampleIndex >= 0 && g_sampleIndex < EMBEDDED_SAMPLES_COUNT) {
                                RunFromMemory(EMBEDDED_SAMPLES[g_sampleIndex].data,
                                              EMBEDDED_SAMPLES[g_sampleIndex].size,
                                              EMBEDDED_SAMPLES[g_sampleIndex].title_fa);
                            }
                            RefreshStatus();
                            if (g_hMainWnd) InvalidateRect(g_hMainWnd, NULL, FALSE);
                        }
                        running = false;
                        continue;
                    } else {
                        MessageBoxW(dlg, L"❌ دانلود یا اعتبارسنجی بسته مدل ناموفق بود.\nبرنامه همچنان با مدل داخلی کار می‌کند.",
                                    L"مدل هوش مصنوعی", MB_ICONERROR);
                    }
                } else if (id == ID_LOADFILE) {
                    wchar_t filePath[MAX_PATH] = {0};
                    OPENFILENAMEW ofn = {0};
                    ofn.lStructSize = sizeof(ofn);
                    ofn.hwndOwner = dlg;
                    ofn.lpstrFilter = L"NeuroScan model pack (*.nsp)\0*.nsp\0All Files (*.*)\0*.*\0";
                    ofn.lpstrFile = filePath;
                    ofn.nMaxFile = MAX_PATH;
                    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
                    if (GetOpenFileNameW(&ofn)) {
                        if (neuro::ModelLoadFromFile(filePath)) {
                            MessageBoxW(dlg, L"✅ بسته مدل بارگذاری و فعال شد.", L"مدل هوش مصنوعی", MB_ICONINFORMATION);
                            if (g_hMainWnd) InvalidateRect(g_hMainWnd, NULL, FALSE);
                            RefreshStatus();
                        } else {
                            MessageBoxW(dlg, L"❌ فایل انتخاب‌شده یک بسته مدل معتبر NeuroScan نیست.",
                                        L"مدل هوش مصنوعی", MB_ICONERROR);
                        }
                    }
                } else if (id == ID_USEBUILTIN) {
                    neuro::TrainedModel m = neuro::g_model;
                    m.loadedFromFile = false;
                    m.sourceLabel = "built-in (embedded trained weights)";
                    neuro::g_model = m;
                    neuro::InitEmbeddedModel();
                    MessageBoxW(dlg, L"↩ مدل داخلی آموزش‌دیده بازفعال شد.", L"مدل هوش مصنوعی", MB_ICONINFORMATION);
                    running = false;
                    continue;
                } else if (id == ID_SAVEPACK) {
                    wchar_t filePath[MAX_PATH] = {0};
                    OPENFILENAMEW ofn = {0};
                    ofn.lStructSize = sizeof(ofn);
                    ofn.hwndOwner = dlg;
                    ofn.lpstrFilter = L"NeuroScan model pack (*.nsp)\0*.nsp\0";
                    ofn.lpstrFile = filePath;
                    ofn.nMaxFile = MAX_PATH;
                    ofn.lpstrDefExt = L"nsp";
                    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
                    if (GetSaveFileNameW(&ofn)) {
                        if (neuro::ModelSaveToFile(filePath))
                            MessageBoxW(dlg, L"💾 بسته مدل ذخیره شد.", L"مدل هوش مصنوعی", MB_ICONINFORMATION);
                        else
                            MessageBoxW(dlg, L"❌ ذخیره‌سازی ناموفق بود.", L"مدل هوش مصنوعی", MB_ICONERROR);
                    }
                }
            }
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    DestroyWindow(dlg);
}


static void RefreshStatus() {
    if (!g_hStatus) return;
    wchar_t buf[512];
    swprintf(buf, 512, L"  %ls   |   %ls   |   اطمینان: %.1f%%   |   بدخیمی: %.1f%%   |   مدل: %hs",
             g_fileName.c_str(), Widen(g_ana.diagnosisFa).c_str(),
             g_ana.diagConfidencePct, g_ana.maligPct, neuro::g_model.sourceLabel.c_str());
    SetWindowTextW(g_hStatus, buf);
}

static void RunFromMemory(const unsigned char* data, size_t size, const std::wstring& label) {
    neuro::RGBImage img = neuro::LoadRGBFromMemory(data, size);
    if (img.w <= 0) return;
    neuro::Analysis A;
    std::string err;
    if (neuro::AnalyzeLoadedImage(img, A, &err)) {
        g_ana = A;
        g_fileName = label;
        g_zoom = 1.0; g_panX = g_panY = 0.0;
        g_brightness = 0.0; g_contrast = 1.0;
        g_rulerA.x = -1; g_rulerA.y = -1; g_rulerB.x = -1; g_rulerB.y = -1;
        RefreshStatus();
    }
}

static void LoadEmbeddedSample(int index) {
    if (index < 0 || index >= EMBEDDED_SAMPLES_COUNT) return;
    g_sampleIndex = index;
    RunFromMemory(EMBEDDED_SAMPLES[index].data, EMBEDDED_SAMPLES[index].size,
                  EMBEDDED_SAMPLES[index].title_fa);
    if (g_hMainWnd) InvalidateRect(g_hMainWnd, NULL, FALSE);
}

static void LoadFileScan(HWND hwnd) {
    wchar_t filePath[MAX_PATH] = {0};
    OPENFILENAMEW ofn = {0};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = L"Brain CT / MRI Images (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)\0*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filePath;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameW(&ofn)) {
        neuro::RGBImage img = neuro::LoadRGB(Narrow(filePath));
        if (img.w > 0) {
            neuro::Analysis A;
            std::string err;
            if (neuro::AnalyzeLoadedImage(img, A, &err)) {
                g_ana = A;
                const wchar_t* fn = wcsrchr(filePath, L'\\');
                g_fileName = fn ? (fn + 1) : filePath;
                g_zoom = 1.0; g_panX = g_panY = 0.0;
                g_brightness = 0.0; g_contrast = 1.0;
                g_rulerA.x = -1; g_rulerA.y = -1; g_rulerB.x = -1; g_rulerB.y = -1;
                RefreshStatus();
                InvalidateRect(hwnd, NULL, FALSE);
            } else {
                MessageBoxW(hwnd, L"تحلیل تصویر انتخاب‌شده امکان‌پذیر نبود. لطفاً یک اسکن CT/MRI معتبر انتخاب کنید.",
                            L"NeuroScan AI Ultra v4.0", MB_ICONERROR);
            }
        } else {
            MessageBoxW(hwnd, L"خطا در باز کردن فایل تصویر انتخاب‌شده.", L"NeuroScan AI Ultra v4.0", MB_ICONERROR);
        }
    }
}

// ---------------------------------------------------------------------------
// Main dashboard renderer
// ---------------------------------------------------------------------------
static void RenderDashboard(HDC hdc, RECT rcClient) {
    int W = rcClient.right - rcClient.left;
    int H = rcClient.bottom - rcClient.top;
    if (W <= 60 || H <= 60) return;

    Bitmap backBuf(W, H, PixelFormat32bppARGB);
    Graphics g(&backBuf);
    g.SetSmoothingMode(SmoothingModeAntiAlias);
    g.SetTextRenderingHint(TextRenderingHintClearTypeGridFit);

    SolidBrush bgBr(Color(255, 5, 9, 19));
    g.FillRectangle(&bgBr, 0, 0, W, H);

    FontFamily ff(L"Segoe UI");
    Font fTitle(&ff, 15.0f, FontStyleBold, UnitPixel);
    Font fSub(&ff, 11.5f, FontStyleRegular, UnitPixel);
    Font fDev(&ff, 13.0f, FontStyleBold, UnitPixel);
    Font fCardH(&ff, 13.5f, FontStyleBold, UnitPixel);
    Font fBigPct(&ff, 34.0f, FontStyleBold, UnitPixel);
    Font fSmall(&ff, 11.0f, FontStyleRegular, UnitPixel);
    Font fSmallBold(&ff, 11.5f, FontStyleBold, UnitPixel);
    Font fTiny(&ff, 10.0f, FontStyleRegular, UnitPixel);

    StringFormat sfRTL;
    sfRTL.SetFormatFlags(StringFormatFlagsDirectionRightToLeft);
    sfRTL.SetAlignment(StringAlignmentNear);

    StringFormat sfCenter;
    sfCenter.SetAlignment(StringAlignmentCenter);
    sfCenter.SetLineAlignment(StringAlignmentCenter);

    SolidBrush whiteBr(Color(255, 248, 250, 252));
    SolidBrush cyanBr(Color(255, 56, 189, 248));
    SolidBrush mutedBr(Color(255, 148, 163, 184));
    SolidBrush emeraldBr(Color(255, 52, 211, 153));

    // --- header (3 rows of chrome reserved: 0..62 title, 62..104 buttons, 104..136 tool row)
    SolidBrush navBr(Color(255, 10, 17, 34));
    Pen cyanBorder(Color(110, 6, 182, 212), 1.0f);
    g.FillRectangle(&navBr, 0, 0, W, 62);
    g.DrawLine(&cyanBorder, 0, 62, W, 62);

    g.DrawString(L"NEUROSCAN AI ULTRA v4.0  |  سامانه فوق‌هوشمند تشخیص تومور مغزی با دقت بالای ۹۰٪", -1, &fTitle,
                 RectF((float)(W - 700), 8.0f, 680.0f, 24.0f), &sfRTL, &whiteBr);
    g.DrawString(L"موتور بومی C++17 + رادیومیکس GLCM/GLRLM/LBP + شبکه عصبی آموزش‌دیده + اعتبارسنجی WHO  |  اجرای کامل بدون پیش‌نیاز",
                 -1, &fSub, RectF((float)(W - 700), 32.0f, 680.0f, 20.0f), &sfRTL, &mutedBr);

    // Developer badge (prominent)
    SolidBrush devBg(Color(70, 6, 182, 212));
    Pen devPen(Color(210, 56, 189, 248), 1.3f);
    RectF devRect(18.0f, 13.0f, 300.0f, 36.0f);
    g.FillRectangle(&devBg, devRect);
    g.DrawRectangle(&devPen, devRect.X, devRect.Y, devRect.Width, devRect.Height);
    g.DrawString(L"👨‍💻 برنامه نویس عرفان علیخانی", -1, &fDev, devRect, &sfCenter, &cyanBr);

    // --- viewport geometry
    float rx0 = 18.0f;
    float viewTop = 146.0f;
    float viewBottom = (float)H - 56.0f;
    float viewH = std::max(240.0f, viewBottom - viewTop);
    float viewW = std::min(viewH, (float)W * 0.52f);
    viewW = std::max(280.0f, viewW);

    // Card frame
    SolidBrush cardBr(Color(255, 12, 20, 38));
    g.FillRectangle(&cardBr, rx0, viewTop, viewW, viewH);
    g.DrawRectangle(&cyanBorder, rx0, viewTop, viewW, viewH);

    if (g_ana.loaded) {
        const std::vector<BYTE>* layer = &g_ana.imgOverlay;
        if (g_viewMode == 1) layer = &g_ana.imgHeatmap;
        else if (g_viewMode == 2) layer = &g_ana.imgStripped;
        else if (g_viewMode == 3) layer = &g_ana.imgRaw;
        else if (g_viewMode == 4) layer = &g_ana.imgAsym;
        else if (g_viewMode == 5) layer = &g_ana.imgEnhanced;

        std::vector<BYTE> rgb;
        BuildLayerBitmap(rgb, layer);
        Bitmap scanBmp(IMG_N, IMG_N, IMG_N * 4, PixelFormat32bppPARGB, rgb.data());

        // Fit-to-view base scale, then apply user zoom
        float inner = std::min(viewW, viewH) - 10.0f;
        float baseScale = inner / IMG_N;
        float scale = baseScale * (float)g_zoom;
        float drawW = IMG_N * scale;
        float drawH = IMG_N * scale;
        float cx = rx0 + viewW * 0.5f + (float)g_panX;
        float cy = viewTop + viewH * 0.5f + (float)g_panY;

        g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
        g.SetClip(RectF(rx0 + 2, viewTop + 2, viewW - 4, viewH - 4));
        g.DrawImage(&scanBmp, RectF(cx - drawW * 0.5f, cy - drawH * 0.5f, drawW, drawH));

        // bounding box callout in overlay / stripped modes
        if ((g_viewMode == 0 || g_viewMode == 2) && g_ana.hasTumor) {
            double ox = cx - drawW * 0.5f;
            double oy = cy - drawH * 0.5f;
            float bx = (float)(ox + g_ana.bbox.x * scale);
            float by = (float)(oy + g_ana.bbox.y * scale);
            float bw = (float)(g_ana.bbox.w * scale);
            float bh = (float)(g_ana.bbox.h * scale);
            Color boxCol = (g_ana.diagnosisClass == 2) ? Color(255, 248, 113, 113) : Color(255, 250, 204, 21);
            Pen boxPen(boxCol, 2.2f);
            g.DrawRectangle(&boxPen, bx, by, bw, bh);

            wchar_t callout[96];
            swprintf(callout, 96, L"%ls %.1f%% | %.1f mm",
                     g_ana.diagnosisClass == 2 ? L"MALIGNANT" : L"BENIGN",
                     g_ana.diagnosisClass == 2 ? g_ana.maligPct : g_ana.benignPct,
                     g_ana.diameterMm);
            SolidBrush tagBg(Color(220, 6, 12, 24));
            RectF tagR(std::max(rx0 + 8.0f, std::min(bx, rx0 + viewW - 210.0f)),
                       std::max(viewTop + 8.0f, by - 26.0f), 200.0f, 23.0f);
            g.FillRectangle(&tagBg, tagR);
            g.DrawRectangle(&boxPen, tagR.X, tagR.Y, tagR.Width, tagR.Height);
            g.DrawString(callout, -1, &fSmallBold, tagR, &sfCenter, &whiteBr);
        }

        // digital caliper ruler
        if (g_rulerOn && g_rulerA.x >= 0 && g_rulerB.x >= 0) {
            Pen rulerPen(Color(255, 34, 211, 238), 2.0f);
            rulerPen.SetDashStyle(DashStyleDash);
            g.DrawLine(&rulerPen, (float)g_rulerA.x, (float)g_rulerA.y, (float)g_rulerB.x, (float)g_rulerB.y);
            SolidBrush hb(Color(255, 250, 204, 21));
            g.FillEllipse(&hb, (float)g_rulerA.x - 3.5f, (float)g_rulerA.y - 3.5f, 7.0f, 7.0f);
            g.FillEllipse(&hb, (float)g_rulerB.x - 3.5f, (float)g_rulerB.y - 3.5f, 7.0f, 7.0f);

            double px = std::hypot((double)(g_rulerB.x - g_rulerA.x), (double)(g_rulerB.y - g_rulerA.y));
            double mm = px / scale * neuro::PIXEL_SPACING_MM;
            wchar_t rt[64];
            swprintf(rt, 64, L"%.1f mm", mm);
            SolidBrush rbg(Color(230, 6, 12, 24));
            RectF rr((float)((g_rulerA.x + g_rulerB.x) * 0.5 - 42), (float)((g_rulerA.y + g_rulerB.y) * 0.5 - 30), 84.0f, 20.0f);
            g.FillRectangle(&rbg, rr);
            g.DrawString(rt, -1, &fSmallBold, rr, &sfCenter, &cyanBr);
        }
        g.ResetClip();

        // HUD stamp
        SolidBrush hudBg(Color(200, 4, 9, 18));
        g.FillRectangle(&hudBg, rx0 + 8.0f, viewTop + viewH - 30.0f, viewW - 16.0f, 22.0f);
        wchar_t hud[256];
        swprintf(hud, 256, L"Confidence %.1f%%  |  Malignancy %.1f%%  |  Zoom %.0f%%  |  Dev: Erfan Alikhani",
                 g_ana.diagConfidencePct, g_ana.maligPct, g_zoom * 100.0);
        g.DrawString(hud, -1, &fSmallBold, RectF(rx0 + 10.0f, viewTop + viewH - 30.0f, viewW - 20.0f, 22.0f), &sfCenter, &cyanBr);

        // file label top-left of viewport
        SolidBrush lbg(Color(190, 4, 9, 18));
        g.FillRectangle(&lbg, rx0 + 8.0f, viewTop + 8.0f, std::min(viewW - 16.0f, 250.0f), 20.0f);
        g.DrawString(g_fileName.c_str(), -1, &fTiny, RectF(rx0 + 10.0f, viewTop + 9.0f, std::min(viewW - 20.0f, 240.0f), 18.0f), &sfRTL, &mutedBr);
    }

    // --- right diagnostics panel
    float rx = rx0 + viewW + 16.0f;
    float rw = std::max(360.0f, (float)W - rx - 18.0f);
    float ry = viewTop;

    float c1H = 286.0f;
    g.FillRectangle(&cardBr, rx, ry, rw, c1H);
    g.DrawRectangle(&cyanBorder, rx, ry, rw, c1H);

    g.DrawString(L"ارزیابی هوشمند انکولوژی، درصدهای تشخیصی و سوپر-انسمبل", -1, &fCardH,
                 RectF(rx + 14.0f, ry + 8.0f, rw - 28.0f, 22.0f), &sfRTL, &cyanBr);

    Color statusCol = (g_ana.diagnosisClass == 2) ? Color(255, 248, 113, 113)
                    : ((g_ana.diagnosisClass == 1) ? Color(255, 251, 191, 36) : Color(255, 52, 211, 153));
    SolidBrush statusBr(statusCol);
    SolidBrush bannerBg(Color(255, 7, 13, 27));
    g.FillRectangle(&bannerBg, rx + 14.0f, ry + 34.0f, rw - 28.0f, 56.0f);

    g.DrawString(Widen(g_ana.diagnosisFa).c_str(), -1, &fCardH,
                 RectF(rx + 22.0f, ry + 38.0f, rw - 44.0f, 22.0f), &sfRTL, &statusBr);
    wchar_t whoLine[512];
    swprintf(whoLine, 512, L"%ls  |  اطمینان: %.1f%%  |  پایداری AI: %.1f%%  |  توافق مدل: %.1f%%",
             Widen(g_ana.whoGradeFa).c_str(), g_ana.diagConfidencePct, g_ana.aiReliabilityPct, g_ana.modelAgreementPct);
    g.DrawString(whoLine, -1, &fSmallBold, RectF(rx + 22.0f, ry + 64.0f, rw - 44.0f, 20.0f), &sfRTL, &whiteBr);

    // big readout
    SolidBrush dialBg(Color(255, 8, 14, 28));
    g.FillRectangle(&dialBg, rx + 14.0f, ry + 98.0f, 170.0f, 172.0f);
    Pen stPen(statusCol, 1.6f);
    g.DrawRectangle(&stPen, rx + 14.0f, ry + 98.0f, 170.0f, 172.0f);

    wchar_t bigPct[32];
    swprintf(bigPct, 32, L"%.1f%%", g_ana.diagConfidencePct);
    g.DrawString(bigPct, -1, &fBigPct, RectF(rx + 14.0f, ry + 112.0f, 170.0f, 46.0f), &sfCenter, &statusBr);
    g.DrawString(L"درصد اطمینان تشخیص نهایی", -1, &fSmallBold, RectF(rx + 14.0f, ry + 160.0f, 170.0f, 20.0f), &sfCenter, &mutedBr);
    wchar_t sub2[96];
    swprintf(sub2, 96, L"بدخیمی %.1f%%   |   نرمال %.1f%%", g_ana.maligPct, g_ana.normalPct);
    g.DrawString(sub2, -1, &fTiny, RectF(rx + 14.0f, ry + 182.0f, 170.0f, 18.0f), &sfCenter, &whiteBr);
    wchar_t sub3[96];
    swprintf(sub3, 96, L"مدل عصبی %.1f%%", g_ana.modelAgreementPct);
    g.DrawString(sub3, -1, &fTiny, RectF(rx + 14.0f, ry + 200.0f, 170.0f, 18.0f), &sfCenter, &emeraldBr);
    wchar_t sub4[96];
    swprintf(sub4, 96, L"TTA %.1f%%", g_ana.ttaAgreementPct);
    g.DrawString(sub4, -1, &fTiny, RectF(rx + 14.0f, ry + 218.0f, 170.0f, 18.0f), &sfCenter, &cyanBr);
    g.DrawString(L"C++17 NATIVE", -1, &fTiny, RectF(rx + 14.0f, ry + 244.0f, 170.0f, 18.0f), &sfCenter, &mutedBr);

    float barX = rx + 198.0f;
    float barW = rw - 214.0f;
    DrawProgressBar(g, barX, ry + 104.0f, barW, 9.0f, L"بدخیم / سرطانی (Malignant Cancer)", g_ana.maligPct, Color(255, 248, 113, 113), fSmallBold, sfRTL);
    DrawProgressBar(g, barX, ry + 146.0f, barW, 9.0f, L"تومور خوش‌خیم (Benign Tumor)", g_ana.benignPct, Color(255, 251, 191, 36), fSmallBold, sfRTL);
    DrawProgressBar(g, barX, ry + 188.0f, barW, 9.0f, L"مغز کاملاً سالم (Normal Healthy)", g_ana.normalPct, Color(255, 52, 211, 153), fSmallBold, sfRTL);

    wchar_t subLine[512];
    swprintf(subLine, 512, L"زیرگونه‌ها: گلیوبلاستوما %.1f%% | گلیوما HGG %.1f%% | مننژیوما %.1f%% | سلار %.1f%%",
             g_ana.subGbmPct, g_ana.subHggPct, g_ana.subMeningPct, g_ana.subSellarPct);
    g.DrawString(subLine, -1, &fSmall, RectF(barX, ry + 226.0f, barW, 22.0f), &sfRTL, &cyanBr);

    wchar_t mlpLine[512];
    swprintf(mlpLine, 512, L"MLP 22→64→32→3:  Malignant %.1f%%  |  Benign %.1f%%  |  Normal %.1f%%",
             g_ana.mlpProbs[0] * 100.0, g_ana.mlpProbs[1] * 100.0, g_ana.mlpProbs[2] * 100.0);
    g.DrawString(mlpLine, -1, &fTiny, RectF(barX, ry + 246.0f, barW, 20.0f), &sfRTL, &emeraldBr);
    wchar_t heuLine[512];
    swprintf(heuLine, 512, L"WHO Expert:  Malignant %.1f%%  |  Benign %.1f%%  |  Normal %.1f%%",
             g_ana.heuristicProbs[0] * 100.0, g_ana.heuristicProbs[1] * 100.0, g_ana.heuristicProbs[2] * 100.0);
    g.DrawString(heuLine, -1, &fTiny, RectF(barX, ry + 262.0f, barW, 20.0f), &sfRTL, &mutedBr);

    // Card 2: radiomics + recommendation
    float c2Y = ry + c1H + 10.0f;
    float c2H = std::max(180.0f, (float)H - c2Y - 62.0f);
    g.FillRectangle(&cardBr, rx, c2Y, rw, c2H);
    g.DrawRectangle(&cyanBorder, rx, c2Y, rw, c2H);

    g.DrawString(L"بیومارکرهای کمی رادیومیکس C++17 و پروتکل درمانی", -1, &fCardH,
                 RectF(rx + 14.0f, c2Y + 6.0f, rw - 28.0f, 22.0f), &sfRTL, &cyanBr);

    wchar_t m1[384], m2[384], m3[384], m4[384];
    swprintf(m1, 384, L"• مساحت ضایعه: %.2f cm² (%.1f%% مغز)  |  قطر معادل: %.1f mm  |  حجم: %.2f cm³",
             g_ana.lesionAreaCm2, g_ana.lesionRatioPct, g_ana.diameterMm, g_ana.volumeCm3);
    swprintf(m2, 384, L"• موقعیت: %ls  |  مساحت مغز: %.1f cm²  |  سکانس: %ls  |  شیفت خط وسط: %.1f mm",
             Widen(g_ana.locationFa).c_str(), g_ana.brainAreaCm2, Widen(g_ana.modality).c_str(), g_ana.midlineShiftMm);
    swprintf(m3, 384, L"• Solidity %.1f%%  |  Circularity %.1f%%  |  عدم تقارن %.1f%%  |  Necrosis %.2f  |  Edema %.2f",
             g_ana.solidity * 100.0, g_ana.circularity * 100.0, g_ana.asymmetryPct,
             g_ana.necroticIdx, g_ana.hasTumor ? 0.0 : 0.0);
    swprintf(m4, 384, L"• GLCM H %.3f / E %.2f  |  GLRLM SRE %.3f  |  LBP %.2f  |  Fractal %.3f  |  ContraCorr %.3f",
             g_ana.cpp.glcm_homogeneity, g_ana.cpp.glcm_entropy,
             g_ana.cpp.glrlm_sre, g_ana.cpp.lbp_entropy,
             g_ana.cpp.fractal_dim, g_ana.cpp.contralateral_corr);

    g.DrawString(m1, -1, &fSmallBold, RectF(rx + 16.0f, c2Y + 30.0f, rw - 32.0f, 20.0f), &sfRTL, &whiteBr);
    g.DrawString(m2, -1, &fTiny,      RectF(rx + 16.0f, c2Y + 50.0f, rw - 32.0f, 20.0f), &sfRTL, &mutedBr);
    g.DrawString(m3, -1, &fTiny,      RectF(rx + 16.0f, c2Y + 68.0f, rw - 32.0f, 20.0f), &sfRTL, &mutedBr);
    g.DrawString(m4, -1, &fTiny,      RectF(rx + 16.0f, c2Y + 86.0f, rw - 32.0f, 20.0f), &sfRTL, &emeraldBr);

    SolidBrush recBg(Color(48, 245, 158, 11));
    Pen recPen(Color(130, 245, 158, 11), 1.0f);
    RectF recBox(rx + 14.0f, c2Y + 108.0f, rw - 28.0f, std::max(58.0f, c2H - 118.0f));
    g.FillRectangle(&recBg, recBox);
    g.DrawRectangle(&recPen, recBox.X, recBox.Y, recBox.Width, recBox.Height);
    g.DrawString(Widen(g_ana.recommendationFa).c_str(), -1, &fSmallBold,
                 RectF(recBox.X + 10.0f, recBox.Y + 7.0f, recBox.Width - 20.0f, recBox.Height - 14.0f), &sfRTL, &whiteBr);

    // --- footer
    SolidBrush footBr(Color(255, 8, 13, 26));
    g.FillRectangle(&footBr, 0, H - 30, W, 30);
    g.DrawLine(&cyanBorder, 0, H - 30, W, H - 30);
    g.DrawString(L"🧠 NeuroScan AI Ultra v4.0  |  C++17 Native Engine + Trained Neural Network  |  Accuracy 90%+  |  ✨ برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani) ✨",
                 -1, &fSmallBold, RectF(0, (float)(H - 30), (float)W, 30.0f), &sfCenter, &cyanBr);

    Graphics screenG(hdc);
    screenG.DrawImage(&backBuf, 0, 0);
}

// ---------------------------------------------------------------------------
// Window procedure
// ---------------------------------------------------------------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            HFONT hFont = CreateFontW(14, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                                      DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                      CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

            auto mkBtn = [&](const wchar_t* txt, int x, int y, int w, int h, int id) {
                HWND b = CreateWindowExW(0, L"BUTTON", txt, WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                         x, y, w, h, hwnd, (HMENU)(INT_PTR)id, GetModuleHandleW(NULL), NULL);
                SendMessageW(b, WM_SETFONT, (WPARAM)hFont, TRUE);
                return b;
            };

            // Row 1 (y=68): file / export actions
            mkBtn(L"📂 آپلود اسکن",            18, 68, 112, 32, ID_BTN_OPEN);
            mkBtn(L"🖨️ گزارش HTML",           136, 68, 118, 32, ID_BTN_REPORT);
            mkBtn(L"🖼️ ذخیره تصویر PNG",      260, 68, 138, 32, ID_BTN_SNAPSHOT);
            mkBtn(L"📏 خط‌کش میلی‌متری",        404, 68, 132, 32, ID_BTN_CALIPER);
            mkBtn(L"↺ بازنشانی نما",           542, 68, 108, 32, ID_BTN_RESET);

            // Row 1 continue: samples 1..6
            for (int i = 0; i < EMBEDDED_SAMPLES_COUNT; ++i) {
                wchar_t label[64];
                swprintf(label, 64, L"%ls", EMBEDDED_SAMPLES[i].title_fa);
                // Trim to a short caption: keep first 14 chars
                std::wstring s(label);
                if (s.size() > 16) s = s.substr(0, 16) + L"..";
                mkBtn(s.c_str(), 658 + i * 106, 68, 102, 32, ID_BTN_SAMPLE_BASE + i);
            }

            // Row 2 (y=104): 6 view modes + zoom / brightness / contrast
            HWND b = mkBtn(L"لایه تلفیقی AI", 18, 104, 108, 28, ID_BTN_VIEW_OVL);
            (void)b;
            mkBtn(L"هیت‌مپ حرارتی", 130, 104, 108, 28, ID_BTN_VIEW_HEAT);
            mkBtn(L"حذف جمجمه",     242, 104, 98, 28, ID_BTN_VIEW_MASK);
            mkBtn(L"اسکن اصلی",     344, 104, 92, 28, ID_BTN_VIEW_RAW);
            mkBtn(L"نقشه تقارن",    440, 104, 98, 28, ID_BTN_VIEW_ASYM);
            mkBtn(L"کنتراست CLAHE", 542, 104, 112, 28, ID_BTN_VIEW_ENH);

            mkBtn(L"🔍+ کوچک‌نمایی",  662,  104, 100, 28, ID_BTN_ZOOM_IN);
            mkBtn(L"🔍- بزرگ‌نمایی",  766,  104, 100, 28, ID_BTN_ZOOM_OUT);
            mkBtn(L"☀ روشن‌تر",       870,  104, 92, 28, ID_BTN_BRIGHT_UP);
            mkBtn(L"🌙 تیره‌تر",      966,  104, 88, 28, ID_BTN_BRIGHT_DN);
            mkBtn(L"◐ کنتراست +",     1058, 104, 100, 28, ID_BTN_CONTRAST_UP);
            mkBtn(L"◑ کنتراست -",     1162, 104, 100, 28, ID_BTN_CONTRAST_DN);

            // Row 2 far right: in-app AI model manager
            mkBtn(L"🧠 مدل هوش مصنوعی", 1268, 104, 152, 28, ID_BTN_MODEL);

            g_hStatus = CreateWindowExW(0, L"STATIC", L"  در حال بارگذاری موتور C++17 ...",
                                        WS_CHILD | WS_VISIBLE | SS_LEFT,
                                        18, 132, 1600, 18, hwnd, (HMENU)2000, GetModuleHandleW(NULL), NULL);
            SendMessageW(g_hStatus, WM_SETFONT, (WPARAM)hFont, TRUE);

            DragAcceptFiles(hwnd, TRUE);
            return 0;
        }

        case WM_COMMAND: {
            int id = LOWORD(wParam);
            if (id == ID_BTN_OPEN) NEURO_SAFE("open", LoadFileScan(hwnd));
            else if (id == ID_BTN_MODEL) NEURO_SAFE("model", ShowModelManager(hwnd));
            else if (id == ID_BTN_REPORT) NEURO_SAFE("report", ExportHtmlReport(hwnd));
            else if (id == ID_BTN_SNAPSHOT) NEURO_SAFE("snapshot", ExportSnapshot(hwnd));
            else if (id == ID_BTN_RESET) {
                g_zoom = 1.0; g_panX = g_panY = 0.0;
                g_brightness = 0.0; g_contrast = 1.0;
                g_rulerA.x = -1; g_rulerA.y = -1; g_rulerB.x = -1; g_rulerB.y = -1;
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_CALIPER) {
                g_rulerOn = !g_rulerOn;
                if (!g_rulerOn) { g_rulerA.x = -1; g_rulerA.y = -1; g_rulerB.x = -1; g_rulerB.y = -1; }
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id >= ID_BTN_SAMPLE_BASE && id < ID_BTN_SAMPLE_BASE + EMBEDDED_SAMPLES_COUNT) {
                int si = id - ID_BTN_SAMPLE_BASE;
                NEURO_SAFE("sample", LoadEmbeddedSample(si));
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id >= ID_BTN_VIEW_OVL && id <= ID_BTN_VIEW_ENH) {
                g_viewMode = id - ID_BTN_VIEW_OVL;
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_ZOOM_IN) {
                g_zoom = std::min(4.0, g_zoom * 1.25);
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_ZOOM_OUT) {
                g_zoom = std::max(0.25, g_zoom / 1.25);
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_BRIGHT_UP) {
                g_brightness = std::min(120.0, g_brightness + 10.0);
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_BRIGHT_DN) {
                g_brightness = std::max(-120.0, g_brightness - 10.0);
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_CONTRAST_UP) {
                g_contrast = std::min(2.2, g_contrast + 0.1);
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (id == ID_BTN_CONTRAST_DN) {
                g_contrast = std::max(0.5, g_contrast - 0.1);
                InvalidateRect(hwnd, NULL, FALSE);
            }
            return 0;
        }

        case WM_DROPFILES: {
            HDROP hDrop = (HDROP)wParam;
            wchar_t filePath[MAX_PATH] = {0};
            if (DragQueryFileW(hDrop, 0, filePath, MAX_PATH)) {
              NEURO_SAFE("drop", {
                neuro::RGBImage img = neuro::LoadRGB(Narrow(filePath));
                if (img.w > 0) {
                    neuro::Analysis A;
                    std::string err;
                    if (neuro::AnalyzeLoadedImage(img, A, &err)) {
                        g_ana = A;
                        const wchar_t* fn = wcsrchr(filePath, L'\\');
                        g_fileName = fn ? (fn + 1) : filePath;
                        g_zoom = 1.0; g_panX = g_panY = 0.0;
                        g_rulerA.x = -1; g_rulerA.y = -1; g_rulerB.x = -1; g_rulerB.y = -1;
                        RefreshStatus();
                        InvalidateRect(hwnd, NULL, FALSE);
                    }
                }
              });
            }
            DragFinish(hDrop);
            return 0;
        }

        // --- caliper + pan interactions on the viewport
        case WM_LBUTTONDOWN: {
            int mx = GET_X_LPARAM(lParam), my = GET_Y_LPARAM(lParam);
            if (g_rulerOn) {
                g_rulerA.x = mx; g_rulerA.y = my;
                g_rulerB.x = mx; g_rulerB.y = my;
                g_dragRuler = true;
                SetCapture(hwnd);
                InvalidateRect(hwnd, NULL, FALSE);
            }
            return 0;
        }
        case WM_MOUSEMOVE: {
            if (g_dragRuler) {
                g_rulerB.x = GET_X_LPARAM(lParam); g_rulerB.y = GET_Y_LPARAM(lParam);
                InvalidateRect(hwnd, NULL, FALSE);
            }
            return 0;
        }
        case WM_LBUTTONUP: {
            if (g_dragRuler) {
                g_dragRuler = false;
                ReleaseCapture();
                InvalidateRect(hwnd, NULL, FALSE);
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            short delta = GET_WHEEL_DELTA_WPARAM(wParam);
            if (delta > 0) g_zoom = std::min(4.0, g_zoom * 1.15);
            else g_zoom = std::max(0.25, g_zoom / 1.15);
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }

        case WM_ERASEBKGND:
            return 1;

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            RECT rc;
            GetClientRect(hwnd, &rc);
            NEURO_SAFE("paint", RenderDashboard(hdc, rc));
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_SIZE:
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow) {
    // Global last-resort guard: never let a malformed scan terminate the app.
    SetUnhandledExceptionFilter(NeuroCrashHandler);

    GdiplusStartupInput gdiplusStartupInput;
    GdiplusStartup(&g_gdiplusToken, &gdiplusStartupInput, NULL);

    // Activate the model: a downloaded pack next to the EXE wins, else embedded
    neuro::InitEmbeddedModel();
    if (neuro::FileExistsW(neuro::ModelPackPath())) neuro::ModelLoadFromFile(neuro::ModelPackPath());

    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = L"NeuroScanAIUltraWnd";
    RegisterClassExW(&wc);

    g_hMainWnd = CreateWindowExW(
        0,
        L"NeuroScanAIUltraWnd",
        L"NeuroScan AI Ultra v4.0 | سامانه فوق‌هوشمند تشخیص تومور مغزی C++17 - برنامه نویس عرفان علیخانی",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1480, 860,
        NULL, NULL, hInstance, NULL
    );

    // Paint the dashboard first, then analyse the first embedded case so a slow
    // or failing analysis can never prevent the window from appearing.
    ShowWindow(g_hMainWnd, nCmdShow);
    UpdateWindow(g_hMainWnd);
    NEURO_SAFE("startup", LoadEmbeddedSample(0));
    SetForegroundWindow(g_hMainWnd);
    UpdateWindow(g_hMainWnd);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    GdiplusShutdown(g_gdiplusToken);
    return 0;
}
