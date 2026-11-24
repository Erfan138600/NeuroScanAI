// ============================================================================
// NeuroScan AI Ultra v4.0 — In-App AI Model Manager (native Win32 / C++17)
// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
//
// • Works 100% offline with the BUILT-IN trained model (no download needed).
// • Optionally downloads an upgraded model pack from a URL you configure,
//   verifies its offset/size, loads it live and persists it next to the EXE.
// • Model pack format (produced by export_model_pack.py):
//       magic      : "NSCANM4\0"      8 bytes
//       flags      : uint32 (1 = contains CNN weights blob)
//       scalerMean : 22 * float32
//       scalerScale: 22 * float32
//       W0 (22x64) B0 (64) W1 (64x32) B1 (32) W2 (32x3) B2 (3)  : float32
//       cnnBytes   : uint32
//       cnnBlob    : raw PyTorch/ONNX state bytes (downloaded for the Python engine)
// ============================================================================
#pragma once
#ifndef NEURO_WIN_MODEL_HPP
#define NEURO_WIN_MODEL_HPP

#ifdef _WIN32
  #include <windows.h>
  #include <urlmon.h>
  #include <shlobj.h>
  #include <sys/stat.h>
#endif
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace neuro {

static const int MOD_DIM = 22, MOD_H1 = 64, MOD_H2 = 32, MOD_OUT = 3;

struct TrainedModel {
    bool loadedFromFile = false;
    std::string sourceLabel = "built-in (embedded trained weights)";
    float mean[MOD_DIM];
    float scale[MOD_DIM];
    float W0[MOD_DIM * MOD_H1], b0[MOD_H1];
    float W1[MOD_H1 * MOD_H2], b1[MOD_H2];
    float W2[MOD_H2 * MOD_OUT], b2[MOD_OUT];
};

static TrainedModel g_model;

// ---------------------------------------------------------------------------
// 3-class forward pass of the trained network (22 -> 64 -> 32 -> 3)
// ---------------------------------------------------------------------------
static void ModelForward(const double* feat22, double outProbs[3]) {
    double x[MOD_DIM];
    for (int i = 0; i < MOD_DIM; ++i)
        x[i] = (feat22[i] - g_model.mean[i]) / (g_model.scale[i] + 1e-9);

    double h1[MOD_H1];
    for (int j = 0; j < MOD_H1; ++j) {
        double s = g_model.b0[j];
        for (int i = 0; i < MOD_DIM; ++i) s += x[i] * g_model.W0[i * MOD_H1 + j];
        h1[j] = s > 0.0 ? s : 0.0;
    }
    double h2[MOD_H2];
    for (int j = 0; j < MOD_H2; ++j) {
        double s = g_model.b1[j];
        for (int i = 0; i < MOD_H1; ++i) s += h1[i] * g_model.W1[i * MOD_H2 + j];
        h2[j] = s > 0.0 ? s : 0.0;
    }
    double logits[MOD_OUT];
    double mx = -1e30;
    for (int k = 0; k < MOD_OUT; ++k) {
        double s = g_model.b2[k];
        for (int i = 0; i < MOD_H2; ++i) s += h2[i] * g_model.W2[i * MOD_OUT + k];
        logits[k] = s;
        if (s > mx) mx = s;
    }
    double sum = 0.0;
    for (int k = 0; k < MOD_OUT; ++k) {
        outProbs[k] = std::exp(logits[k] - mx);
        sum += outProbs[k];
    }
    for (int k = 0; k < MOD_OUT; ++k) outProbs[k] /= (sum + 1e-12);
}

// ---------------------------------------------------------------------------
// Serialisation helpers
// ---------------------------------------------------------------------------
static bool ModelToBuffer(const TrainedModel& m, std::vector<unsigned char>& out) {
    static const char MAGIC[8] = {'N','S','C','A','N','M','4','\0'};
    out.clear();
    out.insert(out.end(), MAGIC, MAGIC + 8);
    uint32_t flags = 1;
    const unsigned char* fp = (const unsigned char*)&flags;
    out.insert(out.end(), fp, fp + 4);
    auto putFloats = [&](const float* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        out.insert(out.end(), b, b + n * sizeof(float));
    };
    const_cast<TrainedModel&>(m);   // read-only access
    putFloats(m.mean, MOD_DIM); putFloats(m.scale, MOD_DIM);
    putFloats(m.W0, MOD_DIM * MOD_H1); putFloats(m.b0, MOD_H1);
    putFloats(m.W1, MOD_H1 * MOD_H2); putFloats(m.b1, MOD_H2);
    putFloats(m.W2, MOD_H2 * MOD_OUT); putFloats(m.b2, MOD_OUT);
    uint32_t cnnBytes = 0;
    out.insert(out.end(), (const unsigned char*)&cnnBytes, (const unsigned char*)&cnnBytes + 4);
    return true;
}

static bool ModelFromBuffer(const std::vector<unsigned char>& buf, TrainedModel& m) {
    size_t need = 8 + 4 + (size_t)(MOD_DIM * 2 + MOD_DIM * MOD_H1 + MOD_H1 +
                                   MOD_H1 * MOD_H2 + MOD_H2 + MOD_H2 * MOD_OUT + MOD_OUT) * sizeof(float);
    if (buf.size() < need) return false;
    if (std::memcmp(buf.data(), "NSCANM4\0", 8) != 0) return false;

    size_t off = 12;
    auto getFloats = [&](float* dst, size_t n) {
        std::memcpy(dst, buf.data() + off, n * sizeof(float));
        off += n * sizeof(float);
    };
    getFloats(m.mean, MOD_DIM); getFloats(m.scale, MOD_DIM);
    getFloats(m.W0, MOD_DIM * MOD_H1); getFloats(m.b0, MOD_H1);
    getFloats(m.W1, MOD_H1 * MOD_H2); getFloats(m.b1, MOD_H2);
    getFloats(m.W2, MOD_H2 * MOD_OUT); getFloats(m.b2, MOD_OUT);
    return true;
}

#ifdef _WIN32
// ---------------------------------------------------------------------------
// Filesystem helpers (model pack lives next to the EXE)
// ---------------------------------------------------------------------------
static std::wstring ExeDirectory() {
    wchar_t path[MAX_PATH] = {0};
    GetModuleFileNameW(NULL, path, MAX_PATH);
    wchar_t* slash = wcsrchr(path, L'\\');
    if (slash) *slash = 0;
    return std::wstring(path);
}

static std::wstring ModelPackPath() {
    return ExeDirectory() + L"\\neuroscan_model_pack_v4.nsp";
}

static bool FileExistsW(const std::wstring& p) {
    struct _stat64 st;
    return _wstat64(p.c_str(), &st) == 0;
}

static bool ModelSaveToFile(const std::wstring& path) {
    std::vector<unsigned char> buf;
    ModelToBuffer(g_model, buf);
    FILE* fp = _wfopen(path.c_str(), L"wb");
    if (!fp) return false;
    fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    return true;
}

static bool ModelLoadFromFile(const std::wstring& path, bool markAsFile = true) {
    FILE* fp = _wfopen(path.c_str(), L"rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > (16 * 1024 * 1024)) { fclose(fp); return false; }
    std::vector<unsigned char> buf((size_t)sz);
    size_t rd = fread(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    if (rd != buf.size()) return false;
    TrainedModel tmp = g_model;
    if (!ModelFromBuffer(buf, tmp)) return false;
    tmp.loadedFromFile = markAsFile;
    tmp.sourceLabel = "external model pack (downloaded)";
    g_model = tmp;
    return true;
}

#ifdef _WIN32
#endif // _WIN32

// ---------------------------------------------------------------------------
// Optional HTTPS download of an upgraded model pack (non-blocking UI feedback)
// ---------------------------------------------------------------------------
struct DownloadProgress {
    HWND hwnd = NULL;
    volatile long done = 0;
    HRESULT hr = S_OK;
    std::wstring url, dest;
};

static void __stdcall DownloadProgressCallback(DWORD, DWORD total, DWORD progress, LPCWSTR, DWORD_PTR ctx) {
    DownloadProgress* dp = (DownloadProgress*)ctx;
    if (dp && dp->hwnd) {
        int pct = (total > 0) ? (int)((double)progress * 100.0 / total) : 0;
        wchar_t buf[128];
        swprintf(buf, 128, L"  ⬇ دریافت مدل ارتقایافته: %d%%", pct);
        SetWindowTextW(dp->hwnd, buf);
    }
}

static DWORD WINAPI DownloadThread(LPVOID param) {
    DownloadProgress* dp = (DownloadProgress*)param;
    HRESULT hr = URLDownloadToFileW(NULL, dp->url.c_str(), dp->dest.c_str(), 0,
                                    (LPBINDSTATUSCALLBACK)DownloadProgressCallback);
    if (SUCCEEDED(hr)) {
        // Verify the pack really is a valid NeuroScan model before using it
        TrainedModel tmp = g_model;
        FILE* fp = _wfopen(dp->dest.c_str(), L"rb");
        std::vector<unsigned char> buf;
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (sz > 0 && sz < 16 * 1024 * 1024) {
                buf.resize((size_t)sz);
                if (fread(buf.data(), 1, buf.size(), fp) != buf.size()) buf.clear();
            }
            fclose(fp);
        }
        if (buf.empty() || !ModelFromBuffer(buf, tmp)) hr = E_FAIL;
    }
    dp->hr = hr;
    dp->done = 1;
    return 0;
}

#endif // _WIN32

} // namespace neuro
#endif // NEURO_WIN_MODEL_HPP
