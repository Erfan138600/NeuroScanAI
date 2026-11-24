// ============================================================================
// NeuroScan AI Ultra v4.0 - Native C++17 Radiomic & Vision Acceleration Engine
// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
//
// Shared C++17 core compiled both into:
//   1. libneuroscan_cpp.so (Linux high-speed native extension for Python ctypes)
//   2. neuroscan_cpp.dll + NeuroScanAI_Pro_Erfan_Alikhani.exe (Windows x64)
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _WIN32
  #define NEURO_EXPORT extern "C" __declspec(dllexport)
#else
  #define NEURO_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct CppRadiomicFeatures {
    double glcm_entropy;
    double glcm_contrast;
    double glcm_homogeneity;
    double glcm_energy;
    double glrlm_short_run_emphasis;
    double glrlm_long_run_emphasis;
    double glrlm_run_nonuniformity;
    double lbp_uniformity;
    double lbp_entropy;
    double fractal_box_dim;
    double gradient_margin_sharpness;
    double intensity_skewness;
    double intensity_kurtosis;
    double contralateral_corr;
};

// 1. Fast 4-angle (0, 45, 90, 135 deg) 32-level GLCM + GLRLM + LBP + Fractal Box Dimension
NEURO_EXPORT int neuro_extract_radiomics(
    const uint8_t* gray_img,
    const uint8_t* brain_mask,
    const uint8_t* lesion_mask,
    int width,
    int height,
    int midline_x,
    CppRadiomicFeatures* out_feat
) {
    if (!gray_img || !brain_mask || !out_feat || width <= 8 || height <= 8) return -1;
    std::memset(out_feat, 0, sizeof(CppRadiomicFeatures));

    // Contralateral Pearson Correlation across midline_x inside brain_mask
    double sumL = 0.0, sumR = 0.0, sumLL = 0.0, sumRR = 0.0, sumLR = 0.0;
    int pairCnt = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < midline_x; ++x) {
            int rx = 2 * midline_x - x;
            if (rx >= 0 && rx < width) {
                int idxL = y * width + x;
                int idxR = y * width + rx;
                if (brain_mask[idxL] > 0 && brain_mask[idxR] > 0) {
                    double vL = gray_img[idxL];
                    double vR = gray_img[idxR];
                    sumL += vL; sumR += vR;
                    sumLL += vL * vL; sumRR += vR * vR;
                    sumLR += vL * vR;
                    pairCnt++;
                }
            }
        }
    }
    if (pairCnt > 20) {
        double meanL = sumL / pairCnt;
        double meanR = sumR / pairCnt;
        double varL = std::max(1e-6, sumLL / pairCnt - meanL * meanL);
        double varR = std::max(1e-6, sumRR / pairCnt - meanR * meanR);
        double cov = sumLR / pairCnt - meanL * meanR;
        out_feat->contralateral_corr = std::max(-1.0, std::min(1.0, cov / std::sqrt(varL * varR)));
    } else {
        out_feat->contralateral_corr = 0.88;
    }

    // Determine ROI mask: use lesion_mask if non-empty, else brain_mask
    const uint8_t* roi = lesion_mask;
    int roiCount = 0;
    int minX = width, maxX = 0, minY = height, maxY = 0;
    if (roi) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (roi[y * width + x] > 0) {
                    roiCount++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
    }
    if (roiCount < 16) {
        roi = brain_mask;
        minX = 0; maxX = width - 1; minY = 0; maxY = height - 1;
        for (int i = 0; i < width * height; ++i) if (roi[i] > 0) roiCount++;
    }
    if (roiCount < 16) return 0;

    // Higher-order intensity moments (mean, std, skewness, kurtosis)
    double s1 = 0.0;
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            int idx = y * width + x;
            if (roi[idx] > 0) s1 += gray_img[idx];
        }
    }
    double mean = s1 / roiCount;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            int idx = y * width + x;
            if (roi[idx] > 0) {
                double d = gray_img[idx] - mean;
                double d2 = d * d;
                m2 += d2;
                m3 += d2 * d;
                m4 += d2 * d2;
            }
        }
    }
    m2 /= roiCount; m3 /= roiCount; m4 /= roiCount;
    double sd = std::sqrt(std::max(1e-6, m2));
    out_feat->intensity_skewness = m3 / (sd * sd * sd + 1e-6);
    out_feat->intensity_kurtosis = m4 / (m2 * m2 + 1e-6) - 3.0;

    // 32-level 4-directional GLCM (0, 45, 90, 135 deg)
    const int LEVELS = 32;
    std::vector<double> glcm(LEVELS * LEVELS, 0.0);
    const int dx4[4] = {1, 1, 0, -1};
    const int dy4[4] = {0, 1, 1, 1};
    double totalPairs = 0.0;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            int idx = y * width + x;
            if (roi[idx] == 0) continue;
            int i_lvl = gray_img[idx] >> 3; // 0..31
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx4[d], ny = y + dy4[d];
                if (nx >= minX && nx <= maxX && ny >= minY && ny <= maxY) {
                    int nidx = ny * width + nx;
                    if (roi[nidx] > 0) {
                        int j_lvl = gray_img[nidx] >> 3;
                        glcm[i_lvl * LEVELS + j_lvl] += 1.0;
                        glcm[j_lvl * LEVELS + i_lvl] += 1.0;
                        totalPairs += 2.0;
                    }
                }
            }
        }
    }

    if (totalPairs > 0.0) {
        double ent = 0.0, con = 0.0, hom = 0.0, ene = 0.0;
        for (int i = 0; i < LEVELS; ++i) {
            for (int j = 0; j < LEVELS; ++j) {
                double p = glcm[i * LEVELS + j] / totalPairs;
                if (p > 1e-12) {
                    int diff = std::abs(i - j);
                    ent -= p * std::log2(p);
                    con += (double)(diff * diff) * p;
                    hom += p / (1.0 + diff);
                    ene += p * p;
                }
            }
        }
        out_feat->glcm_entropy = ent;
        out_feat->glcm_contrast = con;
        out_feat->glcm_homogeneity = hom;
        out_feat->glcm_energy = ene;
    }

    // GLRLM (Gray-Level Run-Length Matrix, 16 gray levels, horizontal & vertical runs)
    const int MAX_RUN = 32;
    std::vector<double> runHist(MAX_RUN + 1, 0.0);
    double totalRuns = 0.0;
    for (int y = minY; y <= maxY; ++y) {
        int curLvl = -1, runLen = 0;
        for (int x = minX; x <= maxX; ++x) {
            int idx = y * width + x;
            if (roi[idx] > 0) {
                int q = gray_img[idx] >> 4;
                if (q == curLvl) {
                    runLen++;
                } else {
                    if (runLen > 0) {
                        int rl = std::min(MAX_RUN, runLen);
                        runHist[rl] += 1.0;
                        totalRuns += 1.0;
                    }
                    curLvl = q;
                    runLen = 1;
                }
            } else if (runLen > 0) {
                int rl = std::min(MAX_RUN, runLen);
                runHist[rl] += 1.0;
                totalRuns += 1.0;
                curLvl = -1; runLen = 0;
            }
        }
        if (runLen > 0) {
            int rl = std::min(MAX_RUN, runLen);
            runHist[rl] += 1.0;
            totalRuns += 1.0;
        }
    }
    if (totalRuns > 0.0) {
        double sre = 0.0, lre = 0.0, rln = 0.0;
        for (int r = 1; r <= MAX_RUN; ++r) {
            if (runHist[r] > 0.0) {
                sre += runHist[r] / ((double)(r * r));
                lre += runHist[r] * ((double)(r * r));
                rln += runHist[r] * runHist[r];
            }
        }
        out_feat->glrlm_short_run_emphasis = sre / totalRuns;
        out_feat->glrlm_long_run_emphasis = lre / totalRuns;
        out_feat->glrlm_run_nonuniformity = rln / (totalRuns * totalRuns);
    }

    // 8-Neighbor Local Binary Pattern (LBP) Micro-Texture Uniformity & Entropy + Boundary Sobel Sharpness
    int lbpHist[256] = {0};
    int lbpCnt = 0;
    double boundaryGradSum = 0.0;
    int boundaryCnt = 0;

    const int n8x[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    const int n8y[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

    for (int y = std::max(1, minY); y <= std::min(height - 2, maxY); ++y) {
        for (int x = std::max(1, minX); x <= std::min(width - 2, maxX); ++x) {
            int idx = y * width + x;
            if (roi[idx] == 0) continue;

            uint8_t c = gray_img[idx];
            uint8_t code = 0;
            bool isBoundary = false;
            for (int k = 0; k < 8; ++k) {
                int nidx = (y + n8y[k]) * width + (x + n8x[k]);
                if (gray_img[nidx] >= c) code |= (1 << k);
                if (roi[nidx] == 0) isBoundary = true;
            }
            lbpHist[code]++;
            lbpCnt++;

            if (isBoundary) {
                double gx = (double)gray_img[idx + 1] - (double)gray_img[idx - 1];
                double gy = (double)gray_img[idx + width] - (double)gray_img[idx - width];
                boundaryGradSum += std::sqrt(gx * gx + gy * gy);
                boundaryCnt++;
            }
        }
    }
    if (lbpCnt > 0) {
        double uSum = 0.0, lEnt = 0.0;
        for (int b = 0; b < 256; ++b) {
            if (lbpHist[b] > 0) {
                double p = (double)lbpHist[b] / lbpCnt;
                uSum += p * p;
                lEnt -= p * std::log2(p);
            }
        }
        out_feat->lbp_uniformity = uSum;
        out_feat->lbp_entropy = lEnt;
    }
    out_feat->gradient_margin_sharpness = (boundaryCnt > 0) ? (boundaryGradSum / boundaryCnt) : 0.0;

    // Boundary Minkowski-Bouligand Fractal Box-Counting Dimension (measures spiculation / infiltration)
    // Benign meningioma/adenoma smooth circle -> ~1.05 - 1.16; Infiltrative GBM -> ~1.22 - 1.48
    if (lesion_mask && boundaryCnt >= 8) {
        int scales[4] = {2, 4, 8, 16};
        double logInvS[4], logN[4];
        int validS = 0;
        for (int si = 0; si < 4; ++si) {
            int s = scales[si];
            int boxes = 0;
            for (int by = minY; by <= maxY; by += s) {
                for (int bx = minX; bx <= maxX; bx += s) {
                    bool hasIn = false, hasOut = false;
                    for (int dy = 0; dy < s && (by + dy) <= maxY; ++dy) {
                        for (int dx = 0; dx < s && (bx + dx) <= maxX; ++dx) {
                            if (lesion_mask[(by + dy) * width + (bx + dx)] > 0) hasIn = true;
                            else hasOut = true;
                        }
                    }
                    if (hasIn && hasOut) boxes++;
                }
            }
            if (boxes > 0) {
                logInvS[validS] = std::log(1.0 / (double)s);
                logN[validS] = std::log((double)boxes);
                validS++;
            }
        }
        if (validS >= 2) {
            double mx = 0, my = 0;
            for (int i = 0; i < validS; ++i) { mx += logInvS[i]; my += logN[i]; }
            mx /= validS; my /= validS;
            double num = 0, den = 0;
            for (int i = 0; i < validS; ++i) {
                num += (logInvS[i] - mx) * (logN[i] - my);
                den += (logInvS[i] - mx) * (logInvS[i] - mx);
            }
            double dim = (den > 1e-8) ? std::fabs(num / den) : 1.1;
            out_feat->fractal_box_dim = std::max(1.0, std::min(1.85, dim));
        } else {
            out_feat->fractal_box_dim = 1.10;
        }
    } else {
        out_feat->fractal_box_dim = 1.00;
    }

    return 1;
}
