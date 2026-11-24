#include "neuro_core.hpp"
#include <cstdio>
using namespace neuro;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "neuroscan-ai/NEUROSCAN-AI-python-project/samples/case_4_normal_healthy_brain.png";
    RGBImage src = LoadRGB(path);
    if (src.w <= 0) { std::printf("load fail\n"); return 1; }
    RGBImage one = IsolateSingleBrain(src);
    GrayImage g = ToGrayScaled(one);
    std::vector<uint8_t> windowed = WindowNormalize(g.px);
    int headArea = 0;
    std::vector<uint8_t> brainMask = SkullStrip(windowed, headArea);
    std::vector<uint8_t> stripped(N * N, 0);
    for (int i = 0; i < N * N; i++) stripped[i] = brainMask[i] ? windowed[i] : 0;
    std::vector<uint8_t> sm = BlurGray(stripped, N, N, 2);
    std::vector<uint8_t> enhanced = CLAHE(sm, N, N, 8, 2.0);
    for (int i = 0; i < N * N; i++) if (!brainMask[i]) enhanced[i] = 0;

    int brainPx = 0; double sumX = 0;
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) if (brainMask[(size_t)y * N + x]) { brainPx++; sumX += x; }
    int midX = (int)std::round(sumX / std::max(1, brainPx));
    std::printf("brainPx=%d area=%.1fcm2 midX=%d\n", brainPx, brainPx * 0.43 * 0.43 / 100.0, midX);

    std::vector<float> enhF(enhanced.begin(), enhanced.end());
    std::vector<float> b21 = BlurFloat(enhF, N, N, 10);
    std::vector<uint8_t> b21u(N * N);
    for (int i = 0; i < N * N; i++) b21u[i] = Clamp255(b21[i]);
    std::vector<uint8_t> mirrored(N * N, 0);
    int span = std::min(midX, N - midX);
    for (int y = 0; y < N; y++) for (int k = 0; k < span; k++) {
        mirrored[(size_t)y * N + (midX - span + k)] = b21u[(size_t)y * N + (midX + span - 1 - k)];
        mirrored[(size_t)y * N + (midX + k)] = b21u[(size_t)y * N + (midX - 1 - k)];
    }
    std::vector<uint8_t> md = MaxFilter(mirrored, N, N, 5);
    std::vector<uint8_t> asymMap(N * N, 0);
    for (int i = 0; i < N * N; i++)
        asymMap[i] = brainMask[i] ? (uint8_t)std::max(0, std::min(255, (int)b21u[i] - (int)md[i])) : 0;

    SegmentResult seg = SegmentLesions(stripped, enhanced, brainMask, asymMap, midX);
    std::printf("lesions=%zu globalAsym=%.3f medRaw=%.1f rawStd=%.1f\n", seg.lesions.size(), seg.globalAsym, seg.medianRaw, seg.robustStd);
    for (auto& L : seg.lesions)
        std::printf("  L area=%.0f bbox=%d,%d,%d,%d sol=%.3f circ=%.3f contrastZ=%.2f meanRaw=%.1f asymMean=%.1f nec=%.2f edema=%.2f\n",
                    L.areaPx, L.bbox.x, L.bbox.y, L.bbox.w, L.bbox.h, L.solidity, L.circularity,
                    L.contrastZ, L.meanRaw, L.asymMean, L.necrosisIdx, L.edemaIdx);

    stbi_write_png("/tmp/d_stripped.png", N, N, 1, stripped.data(), N);
    stbi_write_png("/tmp/d_enhanced.png", N, N, 1, enhanced.data(), N);
    stbi_write_png("/tmp/d_asym.png", N, N, 1, asymMap.data(), N);
    stbi_write_png("/tmp/d_tumor.png", N, N, 1, seg.tumorMask.data(), N);
    return 0;
}
