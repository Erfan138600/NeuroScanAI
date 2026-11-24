"""
NeuroScan AI v3.0 - Clinical Brain CT/MRI Benchmark & Real Patient Case Suite
Developer: عرفان علیخانی (Erfan Alikhani)

Generates calibrated synthetic CT/MRI benchmarks AND cleanly isolates
real uploaded clinical scans (removing artificial clinician circles/boxes/watermarks).
"""

import cv2
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from PIL import Image

from core.cv_engine import BrainCTProcessor


def create_base_brain_ct(size: int = 512) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Synthesizes realistic brain anatomical architecture."""
    np.random.seed(101)
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)

    # 1. Outer scalp & subcutaneous fat
    cv2.ellipse(img, center, (190, 220), 0, 0, 360, 60, -1)

    # 2. Dense skull bone (calvarium)
    cv2.ellipse(img, center, (180, 210), 0, 0, 360, 240, -1)

    # 3. Inner table / Intracranial brain parenchyma
    brain_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(img, center, (168, 198), 0, 0, 360, 110, -1)
    cv2.ellipse(brain_mask, center, (168, 198), 0, 0, 360, 255, -1)

    # Add realistic soft-tissue micro-texture
    noise = np.random.normal(0, 3.2, (size, size)).astype(np.float32)
    brain_floats = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = np.where(brain_mask > 0, brain_floats, img)

    # 4. Symmetrical Lateral Ventricles
    cv2.ellipse(img, (center[0] - 22, center[1] - 25), (10, 32), -15, 0, 360, 32, -1)
    cv2.ellipse(img, (center[0] + 22, center[1] - 25), (10, 32), 15, 0, 360, 32, -1)
    cv2.rectangle(img, (center[0] - 3, center[1] - 10), (center[0] + 3, center[1] + 25), 34, -1)

    # 5. Interhemispheric fissure & cortical sulci
    cv2.line(img, (center[0], center[1] - 195), (center[0], center[1] + 195), 45, 1)
    for rad, angle in [(155, 30), (160, 60), (160, 120), (155, 150),
                       (155, 210), (160, 240), (160, 300), (155, 330)]:
        x1 = int(center[0] + (rad - 15) * np.cos(np.radians(angle)))
        y1 = int(center[1] + (rad - 15) * np.sin(np.radians(angle)))
        x2 = int(center[0] + rad * np.cos(np.radians(angle)))
        y2 = int(center[1] + rad * np.sin(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2, y2), 48, 1)

    return img, brain_mask, center


def generate_glioblastoma_scan(size: int = 512) -> np.ndarray:
    """Case 1: Glioblastoma Multiforme (GBM) - Grade IV Malignant Cancer."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0] + 68, center[1] + 15

    # Vasogenic Edema around tumor
    cv2.ellipse(img, (lx, ly), (60, 68), 25, 0, 360, 75, -1)
    for dx, dy in [(30, 25), (-25, 35), (35, -20), (-30, -30), (0, 45)]:
        cv2.circle(img, (lx + dx, ly + dy), 24, 72, -1)

    # Infiltrative tumor core with irregular lobulated spicules
    pts = []
    num_pts = 16
    for i in range(num_pts):
        angle = i * (2 * np.pi / num_pts)
        r = 38 + 12 * np.sin(3 * angle) + 6 * np.cos(5 * angle)
        px = int(lx + r * np.cos(angle))
        py = int(ly + r * 1.15 * np.sin(angle))
        pts.append([px, py])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 205)

    # Invasive peripheral nodules
    cv2.circle(img, (lx + 26, ly - 20), 16, 215, -1)
    cv2.circle(img, (lx - 24, ly + 26), 18, 210, -1)
    cv2.circle(img, (lx + 32, ly + 18), 17, 220, -1)
    cv2.circle(img, (lx - 12, ly - 28), 14, 205, -1)

    # Central Necrotic / Cystic Core
    cv2.ellipse(img, (lx, ly), (18, 22), 15, 0, 360, 48, -1)

    # Compression of lateral ventricle
    cv2.ellipse(img, (center[0] + 20, center[1] - 25), (3, 14), 15, 0, 360, 50, -1)

    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_meningioma_scan(size: int = 512) -> np.ndarray:
    """Case 2: Benign Meningioma - Grade I."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0] - 96, center[1] - 50

    cv2.circle(img, (lx, ly), 34, 88, -1)
    cv2.circle(img, (lx, ly), 29, 192, -1)

    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_pituitary_adenoma_scan(size: int = 512) -> np.ndarray:
    """Case 3: Pituitary Macroadenoma / Sellar Lesion - Grade I Benign."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0], center[1] - 12
    cv2.ellipse(img, (lx, ly), (22, 25), 0, 0, 360, 188, -1)

    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_normal_brain_scan(size: int = 512) -> np.ndarray:
    """Case 4: Healthy Normal Brain CT Scan."""
    img, brain_mask, center = create_base_brain_ct(size)
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def extract_and_save_real_cases(samples_dir: str) -> List[Dict[str, str]]:
    """
    Processes real clinical MRI/CT files in samples directory using BrainCTProcessor
    (removing clinician red circles/boxes & multi-slice frames) with accurate medical ground-truth labels.
    """
    proc = BrainCTProcessor()

    raw_mappings = [
        {
            "src": "1842-1.avif",
            "dst": "real_case_2_meningioma.png",
            "id": "real_1_glioma_parietal",
            "title_fa": "🔬 بیمار واقعی ۱: گلیومای ارتشاحی پریتو-اکسیپیتال با ادم وسیع (MRI FLAIR)",
            "title_en": "Real Case 1: Infiltrative Parieto-Occipital Glioma",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (2).jpg",
            "dst": "real_case_4_flair_hyperintense.png",
            "id": "real_2_temporal_ring_gbm",
            "title_fa": "🔬 بیمار واقعی ۲: گلیوبلاستوما گیجگاهی با نکروز مرکزی (Ring-Enhancing GBM)",
            "title_en": "Real Case 2: Temporal Ring-Enhancing Glioblastoma",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (3).jpg",
            "dst": "real_case_5_ring_enhancing.png",
            "id": "real_3_multifocal_gbm",
            "title_fa": "🔬 بیمار واقعی ۳: تومور بدخیم عودکننده با رینگ نکروتیک (T1+C Axial)",
            "title_en": "Real Case 3: Recurrent Necrotic High-Grade Glioma",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (1).jpg",
            "dst": "real_case_3_frontal_mass.png",
            "id": "real_4_intraventricular_meningioma",
            "title_fa": "🔬 بیمار واقعی ۴: مننژیوما داخل بطنی هموژن خوش‌خیم (T1+Gadolinium)",
            "title_en": "Real Case 4: Benign Intraventricular Meningioma",
            "diagnosis": "Benign"
        },
        {
            "src": "HH_Case_Study_-_Fig_1_corrected22-1024x475.jpg",
            "dst": "real_case_1_temporal_gbm.png",
            "id": "real_5_hypothalamic_hamartoma",
            "title_fa": "🔬 بیمار واقعی ۵: هامارتوم هیپوتالاموس / توده خوش‌خیم سلار (MRI FLAIR)",
            "title_en": "Real Case 5: Benign Hypothalamic Hamartoma",
            "diagnosis": "Benign"
        },
        {
            "src": "images.jpg",
            "dst": "real_case_6_glioma_axial.png",
            "id": "real_6_healthy_mri",
            "title_fa": "🔬 بیمار واقعی ۶: ام‌آر‌آی مغز کاملاً سالم و طبیعی - سکانس T1 (Normal Brain MRI)",
            "title_en": "Real Case 6: Healthy Normal Brain MRI (T1)",
            "diagnosis": "Normal"
        },
        {
            "src": "images (4).jpg",
            "dst": "real_case_7_cystic_gbm.png",
            "id": "real_7_cystic_necrotic_gbm",
            "title_fa": "🔬 بیمار واقعی ۷: گلیوبلاستوما کیستیک-نکروتیک لوب گیجگاهی با انحراف خط وسط (T1+C)",
            "title_en": "Real Case 7: Cystic-Necrotic Frontotemporal Glioblastoma",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (5).jpg",
            "dst": "real_case_8_healthy_flair.png",
            "id": "real_8_healthy_flair_mri",
            "title_fa": "🔬 بیمار واقعی ۸: ام‌آر‌آی مغز کاملاً سالم و طبیعی - سکانس FLAIR (Normal Brain FLAIR)",
            "title_en": "Real Case 8: Healthy Normal Brain MRI (FLAIR)",
            "diagnosis": "Normal"
        }
    ]

    real_cases = []
    for m in raw_mappings:
        src_path = os.path.join(samples_dir, m["src"])
        dst_path = os.path.join(samples_dir, m["dst"])

        if os.path.exists(src_path):
            try:
                pil_im = Image.open(src_path).convert('RGB')
                arr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                # Special handling for images (1).jpg: right half is T1+C showing crisp enhancing meningioma
                if m["src"] == "images (1).jpg":
                    h, w = arr.shape[:2]
                    arr = arr[:, w // 2:]
                elif m["src"] == "HH_Case_Study_-_Fig_1_corrected22-1024x475.jpg":
                    h, w = arr.shape[:2]
                    arr = arr[:, :w // 2]
                elif m["src"] == "images.jpg":
                    # Top-left slice in the 2x3 healthy brain MRI sheet is T1 Axial (clean, watermark-free)
                    h, w = arr.shape[:2]
                    arr = arr[int(h * 0.05):int(h * 0.52), int(w * 0.03):int(w * 0.36)]
                elif m["src"] == "images (3).jpg":
                    # Bottom-middle slice 'E' showing clear ring-enhancing glioma
                    h, w = arr.shape[:2]
                    arr = arr[int(h * 0.49):int(h * 0.98), int(w * 0.33):int(w * 0.66)]
                elif m["src"] == "images (2).jpg":
                    # Bottom-right axial T1+C slice showing crisp temporal ring-enhancing GBM
                    h, w = arr.shape[:2]
                    arr = arr[int(h * 0.50):, int(w * 0.50):]
                elif m["src"] == "1842-1.avif":
                    # Right axial FLAIR slice showing large parieto-occipital infiltrative glioma
                    h, w = arr.shape[:2]
                    arr = arr[:int(h * 0.88), int(w * 0.50):]
                elif m["src"] == "images (4).jpg":
                    # Bottom-row 2nd slice showing prominent cystic-necrotic temporal GBM with bright medial rim & ventricular compression
                    h, w = arr.shape[:2]
                    arr = arr[int(h * 0.49):int(h * 0.98), int(w * 0.24):int(w * 0.50)]
                elif m["src"] == "images (5).jpg":
                    # Bottom-row 2nd slice showing clean symmetrical normal FLAIR axial brain with lateral ventricles
                    h, w = arr.shape[:2]
                    arr = arr[int(h * 0.50):int(h * 0.97), int(w * 0.16):int(w * 0.48)]

                single_512 = proc.isolate_single_brain(arr)
                cv2.imwrite(dst_path, single_512)

                real_cases.append({
                    "id": m["id"],
                    "filename": m["dst"],
                    "filepath": dst_path,
                    "title_fa": m["title_fa"],
                    "title_en": m["title_en"],
                    "diagnosis": m["diagnosis"]
                })
            except Exception as e:
                print(f"Error processing real case {m['src']}: {e}")

    return real_cases


def generate_all_samples(output_dir: str = "samples") -> List[Dict[str, str]]:
    """Generates calibrated synthetic benchmarks and standardized real clinical cases."""
    os.makedirs(output_dir, exist_ok=True)

    synth_cases = [
        {
            "id": "case_1_gbm_malignant",
            "filename": "case_1_glioblastoma_malignant.png",
            "title_fa": "📐 بنچمارک ۱: گلیوبلاستوما مولتی‌فرم (بدخیم / سرطانی Grade IV)",
            "title_en": "Benchmark 1: Glioblastoma Multiforme (GBM)",
            "diagnosis": "Malignant",
            "generator": generate_glioblastoma_scan
        },
        {
            "id": "case_2_meningioma_benign",
            "filename": "case_2_meningioma_benign.png",
            "title_fa": "📐 بنچمارک ۲: مننژیوما هموژن (تومور خوش‌خیم Grade I)",
            "title_en": "Benchmark 2: Benign Meningioma",
            "diagnosis": "Benign",
            "generator": generate_meningioma_scan
        },
        {
            "id": "case_3_adenoma_benign",
            "filename": "case_3_pituitary_adenoma.png",
            "title_fa": "📐 بنچمارک ۳: ماکروآدنوم هیپوفیز (تومور خوش‌خیم Grade I)",
            "title_en": "Benchmark 3: Pituitary Macroadenoma",
            "diagnosis": "Benign",
            "generator": generate_pituitary_adenoma_scan
        },
        {
            "id": "case_4_normal_healthy",
            "filename": "case_4_normal_healthy_brain.png",
            "title_fa": "📐 بنچمارک ۴: سی‌تی اسکن مغز کاملاً سالم و نرمال",
            "title_en": "Benchmark 4: Normal Healthy Brain CT",
            "diagnosis": "Normal",
            "generator": generate_normal_brain_scan
        }
    ]

    all_cases = []
    real_cases = extract_and_save_real_cases(output_dir)
    all_cases.extend(real_cases)

    for case in synth_cases:
        filepath = os.path.join(output_dir, case["filename"])
        img_bgr = case["generator"](512)
        cv2.imwrite(filepath, img_bgr)
        all_cases.append({
            "id": case["id"],
            "filename": case["filename"],
            "filepath": filepath,
            "title_fa": case["title_fa"],
            "title_en": case["title_en"],
            "diagnosis": case["diagnosis"]
        })

    return all_cases


if __name__ == "__main__":
    cases = generate_all_samples("samples")
    print(f"Loaded and generated {len(cases)} total brain cases (Real & Benchmark).")
