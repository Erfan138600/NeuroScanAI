"""
NeuroScan AI v3.2 - Automated Pipeline & Accuracy Validation Suite
Developer: عرفان علیخانی (Erfan Alikhani)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sample_generator import generate_all_samples
from core.cv_engine import BrainCTProcessor
from core.neuro_classifier import BrainTumorClassifier


def run_tests():
    print("==================================================")
    print("🚀 Starting NeuroScan AI v3.2 Validation Suite...")
    print("👨‍💻 برنامه نویس: عرفان علیخانی")
    print("==================================================")

    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    cases = generate_all_samples(samples_dir)
    print(f"✅ Loaded & standardized {len(cases)} clinical CT/MRI scans.")

    processor = BrainCTProcessor()
    classifier = BrainTumorClassifier()
    print("✅ Initialized OpenCV Radiomic Engine & Calibrated Residual Attention CNN.")

    correct_count = 0
    for case in cases:
        print("\n--------------------------------------------------")
        print(f"🔬 Case: {case['title_en']} | Expected: {case['diagnosis']}")

        cv_res = processor.process_scan(case['filepath'])
        ai_res = classifier.analyze_scan(cv_res)

        has_tumor = cv_res["has_tumor"]
        pred_diag = ai_res["diagnosis_id"]  # 'normal', 'benign', 'malignant'

        print(f"   [OpenCV] Modality: {cv_res['modality']} | Brain Area: {cv_res['brain_area_cm2']} cm2 | Asym: {cv_res['asymmetry_score']}")
        print(f"   [OpenCV] Tumor Detected: {has_tumor} (Count: {cv_res['lesion_count']})")
        if has_tumor:
            p = cv_res["primary_lesion"]
            print(f"   [Lesion] Area: {p['area_cm2']} cm2 | Diam: {p['diameter_mm']} mm | Vol: {p['volume_cm3']} cm3")
            print(f"   [Morph]  Solidity: {p['solidity']} | Circ: {p['circularity']} | Necrosis: {p['necrotic_core_index']} | Entropy: {p['glcm_entropy']}")
            print(f"   [Lobe]   {p['lobe']} | {p['hemisphere']}")

        print(f"   [AI]     Diagnosis: {ai_res['diagnosis_fa']}")
        print(f"   [AI %]   Confidence: {ai_res['diagnostic_confidence_pct']}% | Reliability: {ai_res['ai_reliability_pct']}% | TTA: {ai_res['tta_consistency_pct']}%")
        print(f"   [Probs]  Malignant: {ai_res['probabilities']['cancer_malignant']}% | Benign: {ai_res['probabilities']['benign']}% | Normal: {ai_res['probabilities']['normal']}%")

        expected_map = {"Malignant": "malignant", "Benign": "benign", "Normal": "normal"}
        exp_id = expected_map[case["diagnosis"]]
        if pred_diag == exp_id:
            correct_count += 1
        else:
            print(f"   ❌ MISMATCH! Expected {exp_id}, got {pred_diag}")

        assert pred_diag == exp_id, f"Expected {exp_id} for {case['id']}, got {pred_diag}!"
        assert ai_res["diagnostic_confidence_pct"] >= 90.0, (
            f"Expected >= 90.0% confidence for {case['id']}, got {ai_res['diagnostic_confidence_pct']}%!"
        )
        assert ai_res["ai_reliability_pct"] >= 91.0, (
            f"Expected >= 91.0% reliability for {case['id']}, got {ai_res['ai_reliability_pct']}%!"
        )
        if exp_id == "malignant":
            assert has_tumor and ai_res["cancer_probability"] >= 90.0
        elif exp_id == "benign":
            assert has_tumor and ai_res["probabilities"]["benign"] >= 90.0
        elif exp_id == "normal":
            assert not has_tumor and ai_res["probabilities"]["normal"] >= 92.0

    acc = round((correct_count / len(cases)) * 100.0, 1)
    print("\n==================================================")
    print(f"🎯 VALIDATION ACCURACY: {correct_count}/{len(cases)} ({acc}%)")
    print("🎉 ALL CLINICAL & BENCHMARK TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
