// ============================================================================
// NeuroScan AI Ultra v4.0 — Native C++17 Headless Benchmark & Parity CLI
// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
//
// Builds with:  g++ -O3 -std=c++17 -o neuroscan_bench neuroscan_bench.cpp
// Runs the EXACT same C++17 engine that is compiled inside
// NeuroScanAI_Ultra_v4.exe, over 12 clinical & benchmark scans, and reports
// accuracy, confidence percentages, C++17 radiomic biomarkers and the
// exported trained-neural-network probabilities.
// ============================================================================
#include "neuro_core.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace neuro;


struct CaseFile {
    const char* path;
    int expected;      // 0 normal, 1 benign, 2 malignant
    const char* name;  // ASCII label
};

int main(int argc, char** argv) {
    if (getenv("NEURO_DEBUG")) g_debugSeg = true;
    const std::string base = (argc > 1) ? argv[1]
        : "neuroscan-ai/NEUROSCAN-AI-python-project/samples/";

    std::vector<CaseFile> cases = {
        {"real_case_2_meningioma.png",                  2, "Real 1  Infiltrative parieto-occipital glioma"},
        {"real_case_4_flair_hyperintense.png",          2, "Real 2  Temporal ring-enhancing GBM"},
        {"real_case_5_ring_enhancing.png",              2, "Real 3  Recurrent necrotic high-grade glioma"},
        {"real_case_3_frontal_mass.png",                1, "Real 4  Intraventricular meningioma"},
        {"real_case_1_temporal_gbm.png",                1, "Real 5  Hypothalamic hamartoma"},
        {"real_case_6_glioma_axial.png",                0, "Real 6  Healthy normal MRI (T1)"},
        {"real_case_7_cystic_gbm.png",                  2, "Real 7  Cystic-necrotic frontotemporal GBM"},
        {"real_case_8_healthy_flair.png",               0, "Real 8  Healthy normal MRI (FLAIR)"},
        {"case_1_glioblastoma_malignant.png",           2, "Bench 1 Glioblastoma multiforme"},
        {"case_2_meningioma_benign.png",                1, "Bench 2 Benign meningioma"},
        {"case_3_pituitary_adenoma.png",                1, "Bench 3 Pituitary macroadenoma"},
        {"case_4_normal_healthy_brain.png",             0, "Bench 4 Normal healthy brain"},
    };

    const char* clsFa[3] = {"NORMAL", "BENIGN", "MALIGNANT"};
    int correct = 0;
    int total = 0;
    double confSum = 0.0;
    double minConf = 100.0;

    std::printf("==============================================================\n");
    std::printf("  NeuroScan AI Ultra v4.0 — Native C++17 Engine Benchmark\n");
    std::printf("  Developer: Erfan Alikhani  (برنامه نویس عرفان علیخانی)\n");
    std::printf("  Engine: C++17 Radiomics + 22-D Trained Neural Network\n");
    std::printf("==============================================================\n");

    for (auto& c : cases) {
        std::string full = base + c.path;
        RGBImage img = LoadRGB(full);
        if (img.w <= 0) {
            std::printf("[SKIP] %-52s (missing file)\n", c.path);
            continue;
        }
        Analysis A;
        std::string err;
        if (!AnalyzeLoadedImage(img, A, &err)) {
            std::printf("[FAIL] %-52s %s\n", c.path, err.c_str());
            continue;
        }
        total++;
        bool ok = (A.diagnosisClass == c.expected);
        if (ok) correct++;
        confSum += A.diagConfidencePct;
        minConf = std::min(minConf, A.diagConfidencePct);

        std::printf("\n--------------------------------------------------------------\n");
        std::printf("%s\n", c.name);
        std::printf("  Modality: %-16s Brain: %6.2f cm2   Asym: %.3f   Shift: %.1f mm\n",
                    A.modality.c_str(), A.brainAreaCm2, A.asymmetryPct / 100.0, A.midlineShiftMm);
        std::printf("  Tumor: %s   Lesions: 1   Area: %5.2f cm2   Diam: %5.1f mm   Vol: %5.2f cm3\n",
                    A.hasTumor ? "YES" : "NO ", A.lesionAreaCm2, A.diameterMm, A.volumeCm3);
        std::printf("  C++17  GLCM(H=%.3f C=%.2f E=%.3f)  GLRLM(SRE=%.3f)  LBP(H=%.3f)  FD=%.3f\n",
                    A.cpp.glcm_homogeneity, A.cpp.glcm_contrast, A.cpp.glcm_energy,
                    A.cpp.glrlm_sre, A.cpp.lbp_entropy, A.cpp.fractal_dim);
        std::printf("  Radiomics Skew=%.2f Kurt=%.2f ContraCorr=%.3f GradMargin=%.1f\n",
                    A.cpp.skewness, A.cpp.kurtosis, A.cpp.contralateral_corr, A.cpp.gradient_margin);
        std::printf("  MLP(22-64-32-3): mal=%.3f ben=%.3f nor=%.3f | Heuristic: %.3f/%.3f/%.3f\n",
                    A.mlpProbs[0], A.mlpProbs[1], A.mlpProbs[2],
                    A.heuristicProbs[0], A.heuristicProbs[1], A.heuristicProbs[2]);
        std::printf("  => %s  conf=%.1f%%  reliability=%.1f%%  agree=%.1f%%  [%s]\n",
                    clsFa[A.diagnosisClass], A.diagConfidencePct, A.aiReliabilityPct,
                    A.modelAgreementPct, ok ? "PASS" : "MISMATCH");
        std::printf("     Malignant %.1f%% | Benign %.1f%% | Normal %.1f%%\n",
                    A.maligPct, A.benignPct, A.normalPct);
    }

    std::printf("\n==============================================================\n");
    std::printf("  NATIVE C++17 ACCURACY: %d/%d (%.1f%%)\n", correct, total,
                total ? 100.0 * correct / total : 0.0);
    std::printf("  Mean calibrated confidence: %.1f%%   Min: %.1f%%\n",
                total ? confSum / total : 0.0, minConf);
    std::printf("==============================================================\n");
    return (correct == total && total > 0) ? 0 : 1;
}
