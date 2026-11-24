"""
Generates C++ headers:
  1. `embedded_samples.h`: 6 compressed clinical & benchmark CT/MRI scans (4 real clinical + 2 benchmarks)
  2. `cpp_mlp_weights.h`: Trained C++17 2-layer Neural Network weights + StandardScaler parameters
     exported directly from our Python Super-Ensemble so the standalone Windows .exe runs genuine
     trained AI inference + C++17 4-angle GLCM/GLRLM/LBP/Fractal Radiomics offline!
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("neuroscan-ai/NEUROSCAN-AI-python-project"))
from core.sample_generator import generate_all_samples
from core.neuro_classifier import BrainTumorClassifier

cases = generate_all_samples("neuroscan-ai/NEUROSCAN-AI-python-project/samples")

# 6 rich built-in clinical & benchmark cases for the Windows EXE
selected_ids = [
    ("real_1_glioma_parietal", "۱. گلیومای بدخیم ارتشاحی (Real MRI)"),
    ("real_2_temporal_ring_gbm", "۲. گلیوبلاستوما نکروتیک گیجگاهی (Real GBM)"),
    ("real_4_intraventricular_meningioma", "۳. مننژیوما خوش‌خیم داخل بطنی (Real Benign)"),
    ("real_6_healthy_mri", "۴. ام‌آر‌آی مغز کاملاً سالم (Real Normal)"),
    ("case_1_gbm_malignant", "۵. بنچمارک گلیوبلاستوما Grade IV"),
    ("case_3_adenoma_benign", "۶. بنچمارک آدنوم خوش‌خیم هیپوفیز"),
]

id_to_case = {c["id"]: c for c in cases}

lines = [
    "#pragma once",
    "#include <cstdint>",
    "#include <cstddef>",
    "",
    "struct EmbeddedSample {",
    "    const wchar_t* title_fa;",
    "    const unsigned char* data;",
    "    size_t size;",
    "};",
    ""
]

entries = []
for idx, (cid, label_fa) in enumerate(selected_ids):
    c = id_to_case[cid]
    img = cv2.imread(c["filepath"], cv2.IMREAD_COLOR)
    img_small = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_small, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    raw_bytes = buf.tobytes()
    arr_name = f"SAMPLE_JPG_{idx}"
    hex_vals = ", ".join(f"0x{b:02X}" for b in raw_bytes)
    lines.append(f"static const unsigned char {arr_name}[{len(raw_bytes)}] = {{ {hex_vals} }};")
    entries.append(f'    {{ L"{label_fa}", {arr_name}, {len(raw_bytes)} }}')

lines.append("")
lines.append(f"static const EmbeddedSample EMBEDDED_SAMPLES[{len(entries)}] = {{")
lines.append(",\n".join(entries))
lines.append("};")
lines.append(f"static const int EMBEDDED_SAMPLES_COUNT = {len(entries)};")
lines.append("")

out_samples = "neuroscan-ai/windows-native-exe/embedded_samples.h"
with open(out_samples, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Wrote {out_samples} ({os.path.getsize(out_samples) // 1024} KB)")

# Export trained MLP + Scaler weights from BrainTumorClassifier to C++ header
clf = BrainTumorClassifier()
scaler = clf.scaler
mlp = clf.mlp

mean_arr = scaler.mean_.astype(np.float64)
scale_arr = scaler.scale_.astype(np.float64)
W0 = mlp.coefs_[0].astype(np.float64)   # (22, 64)
b0 = mlp.intercepts_[0].astype(np.float64)  # (64,)
W1 = mlp.coefs_[1].astype(np.float64)   # (64, 32)
b1 = mlp.intercepts_[1].astype(np.float64)  # (32,)
W2 = mlp.coefs_[2].astype(np.float64)   # (32, 3)
b2 = mlp.intercepts_[2].astype(np.float64)  # (3,)

def fmt_arr(name: str, arr: np.ndarray) -> str:
    flat = arr.ravel()
    vals = ", ".join(f"{v:.8f}" for v in flat)
    return f"static const double {name}[{len(flat)}] = {{ {vals} }};"

w_lines = [
    "#pragma once",
    "// Auto-exported Calibrated Neural Network (22 -> 64 -> 32 -> 3) Weights for Native C++17 Inference",
    "// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)",
    "#include <cmath>",
    "#include <algorithm>",
    "",
    f"static const int CPP_RAD_DIM = {len(mean_arr)};",
    f"static const int CPP_H1_DIM = {len(b0)};",
    f"static const int CPP_H2_DIM = {len(b1)};",
    f"static const int CPP_OUT_DIM = {len(b2)};",
    "",
    fmt_arr("CPP_SCALER_MEAN", mean_arr),
    fmt_arr("CPP_SCALER_SCALE", scale_arr),
    fmt_arr("CPP_MLP_W0", W0),
    fmt_arr("CPP_MLP_B0", b0),
    fmt_arr("CPP_MLP_W1", W1),
    fmt_arr("CPP_MLP_B1", b1),
    fmt_arr("CPP_MLP_W2", W2),
    fmt_arr("CPP_MLP_B2", b2),
    "",
    "inline void RunTrainedCppMLP(const double* feat22, double out_probs[3]) {",
    "    double x[CPP_RAD_DIM];",
    "    for (int i = 0; i < CPP_RAD_DIM; ++i) {",
    "        x[i] = (feat22[i] - CPP_SCALER_MEAN[i]) / (CPP_SCALER_SCALE[i] + 1e-9);",
    "    }",
    "    double h1[CPP_H1_DIM];",
    "    for (int j = 0; j < CPP_H1_DIM; ++j) {",
    "        double s = CPP_MLP_B0[j];",
    "        for (int i = 0; i < CPP_RAD_DIM; ++i) s += x[i] * CPP_MLP_W0[i * CPP_H1_DIM + j];",
    "        h1[j] = (s > 0.0) ? s : 0.0;",
    "    }",
    "    double h2[CPP_H2_DIM];",
    "    for (int j = 0; j < CPP_H2_DIM; ++j) {",
    "        double s = CPP_MLP_B1[j];",
    "        for (int i = 0; i < CPP_H1_DIM; ++i) s += h1[i] * CPP_MLP_W1[i * CPP_H2_DIM + j];",
    "        h2[j] = (s > 0.0) ? s : 0.0;",
    "    }",
    "    double logits[3];",
    "    double maxL = -1e9;",
    "    for (int k = 0; k < 3; ++k) {",
    "        double s = CPP_MLP_B2[k];",
    "        for (int i = 0; i < CPP_H2_DIM; ++i) s += h2[i] * CPP_MLP_W2[i * 3 + k];",
    "        logits[k] = s;",
    "        if (s > maxL) maxL = s;",
    "    }",
    "    double sumExp = 0.0;",
    "    for (int k = 0; k < 3; ++k) {",
    "        out_probs[k] = std::exp(logits[k] - maxL);",
    "        sumExp += out_probs[k];",
    "    }",
    "    for (int k = 0; k < 3; ++k) out_probs[k] /= (sumExp + 1e-12);",
    "}",
    ""
]

# ---------------------------------------------------------------------------
# Also export the binary "model pack" (.nsp) that the Windows EXE can load at
# runtime / download from the in-app Model Manager.  Same layout as
# neuro_win_model.hpp :: ModelFromBuffer()
# ---------------------------------------------------------------------------
import struct
model_pack = bytearray()
model_pack += b"NSCANM4\x00"                       # magic (8 bytes)
model_pack += struct.pack("<I", 1)                 # flags: 1 = trained weights present
for arr in (mean_arr, scale_arr, W0, b0, W1, b1, W2, b2):
    model_pack += np.asarray(arr, dtype=np.float32).ravel().tobytes()
model_pack += struct.pack("<I", 0)                 # cnn blob length (0 = MLP only)
with open("neuroscan-ai/windows-native-exe/neuroscan_model_pack_v4.nsp", "wb") as f:
    f.write(model_pack)
print(f"Wrote neuroscan_model_pack_v4.nsp ({len(model_pack) // 1024} KB)")

out_weights = "neuroscan-ai/windows-native-exe/cpp_mlp_weights.h"
with open(out_weights, "w", encoding="utf-8") as f:
    f.write("\n".join(w_lines))
print(f"Wrote {out_weights} ({os.path.getsize(out_weights) // 1024} KB)")
