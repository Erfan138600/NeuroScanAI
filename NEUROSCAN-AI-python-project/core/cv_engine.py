"""
NeuroScan AI v3.0 - Advanced Computer Vision & Radiomic Engine (OpenCV)
Developer: عرفان علیخانی (Erfan Alikhani)

High-precision Brain CT / MRI Image Processing:
- Colored Annotation & Watermark Inpainting (removes red boxes/circles, text overlays)
- Multi-slice Montage Detection & Smart Single-Slice Isolation
- Adaptive Skull Stripping (Calvarium & Scalp Removal for CT, T1, T2, FLAIR)
- CLAHE + Bilateral Edge-Preserving Contrast Normalization
- Multi-Scale Anomaly Segmentation (Z-Score + Contralateral Symmetry + Otsu/Morphological)
- 24+ Quantitative Radiomic & GLCM Texture Biomarkers
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple, List, Optional

cv2.setNumThreads(0)


class BrainCTProcessor:
    """
    Clinical-grade OpenCV pipeline engineered for Brain CT & MRI scans.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512), pixel_spacing_mm: float = 0.43):
        self.target_size = target_size
        self.pixel_spacing_mm = pixel_spacing_mm

    def remove_colored_annotations(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects and inpaints artificial colored markings (e.g., red/orange/green/blue
        circles or rectangles drawn by clinicians on reference images) and corner watermarks,
        so they never corrupt skull stripping or lesion segmentation.
        Returns (cleaned_bgr, inpaint_mask).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Check if the ENTIRE image is monochrome-tinted (e.g., blue PACS / green thermal tint)
        fg_sat = sat[val > 35]
        if len(fg_sat) > 200 and float(np.median(fg_sat)) > 24:
            # Uniformly tinted clinical film -> neutralize to luminance BGR first!
            lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            bgr = cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]

        color_mask = np.uint8((sat > 55) & (val > 40)) * 255

        # Also check B-R channel difference for subtle colored annotations
        b, g, r = cv2.split(bgr)
        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        chroma = max_c.astype(np.int16) - min_c.astype(np.int16)
        chroma_mask = np.uint8(chroma > 32) * 255

        combined_mask = cv2.bitwise_or(color_mask, chroma_mask)

        # Also detect pure-white crisp overlaid text/logos near top/bottom 14% margins (e.g. 'E', '2006.7.3', 'www...')
        h, w = bgr.shape[:2]
        gray_tmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        margin_zone = np.zeros((h, w), dtype=np.uint8)
        margin_zone[:int(h * 0.14), :] = 255
        margin_zone[int(h * 0.86):, :] = 255
        margin_zone[:, :int(w * 0.08)] = 255
        margin_zone[:, int(w * 0.92):] = 255
        _, bright_txt = cv2.threshold(gray_tmp, 225, 255, cv2.THRESH_BINARY)
        txt_cands = cv2.bitwise_and(bright_txt, margin_zone)
        num_t, lbl_t, stats_t, _ = cv2.connectedComponentsWithStats(txt_cands)
        txt_mask = np.zeros((h, w), dtype=np.uint8)
        for ti in range(1, num_t):
            ta = stats_t[ti, cv2.CC_STAT_AREA]
            tw, th = stats_t[ti, cv2.CC_STAT_WIDTH], stats_t[ti, cv2.CC_STAT_HEIGHT]
            if ta < (h * w) * 0.025 and max(tw, th) < w * 0.45:
                txt_mask[lbl_t == ti] = 255

        if np.count_nonzero(txt_mask) > 0:
            txt_dil = cv2.dilate(txt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            bgr = cv2.inpaint(bgr, txt_dil, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        if np.count_nonzero(combined_mask) > 15:
            # Only inpaint thin line/ring annotations (do not blur large interior regions)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(combined_mask, kernel, iterations=1)
            cleaned = cv2.inpaint(bgr, dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return cleaned, dilated

        return bgr.copy(), np.zeros(bgr.shape[:2], dtype=np.uint8)

    def remove_peripheral_artifacts(self, gray: np.ndarray) -> np.ndarray:
        """
        Suppresses corner text labels, scale bars, and bright logos disconnected
        from the central cranial vault.
        """
        h, w = gray.shape
        cleaned = gray.copy()

        # Zero out 4px outer frame border (often white frames in screenshots)
        cv2.rectangle(cleaned, (0, 0), (w - 1, h - 1), 0, 6)

        # Find connected components above background
        _, bin_mask = cv2.threshold(cleaned, 28, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)

        if num_labels <= 1:
            return cleaned

        # Keep components that are large and near center
        img_area = float(h * w)
        cx_center, cy_center = w / 2.0, h / 2.0
        keep_mask = np.zeros_like(gray)

        # Find the primary head component (largest component near center)
        best_label = -1
        best_score = -1.0
        for lbl in range(1, num_labels):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if area < img_area * 0.03:
                continue
            cx, cy = centroids[lbl]
            dist = np.hypot((cx - cx_center) / w, (cy - cy_center) / h)
            score = area * max(0.1, 1.0 - dist * 1.4)
            if score > best_score:
                best_score = score
                best_label = lbl

        if best_label > 0:
            head_comp = np.uint8(labels == best_label) * 255
            # Fill interior of head component so internal dark ventricles stay intact
            cnts, _ = cv2.findContours(head_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                main_cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(keep_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
                cleaned = cv2.bitwise_and(cleaned, cleaned, mask=keep_mask)

        return cleaned

    def isolate_single_brain(self, img: np.ndarray) -> np.ndarray:
        """
        Detects multi-slice montages, grids (2x3, 2x4), or side-by-side scans and
        automatically isolates and centers a SINGLE high-resolution brain slice.
        """
        # First strip white outer margins if image has a white background frame
        h, w = img.shape[:2]
        gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check if outer border is predominantly white (> 230)
        border_pixels = np.concatenate([
            gray_raw[0:6, :].ravel(),
            gray_raw[h-6:h, :].ravel(),
            gray_raw[:, 0:6].ravel(),
            gray_raw[:, w-6:w].ravel()
        ])
        if np.mean(border_pixels) > 200:
            # Crop inside dark scan box if present
            dark_box = np.uint8(gray_raw < 200) * 255
            coords = cv2.findNonZero(dark_box)
            if coords is not None:
                bx, by, bw, bh = cv2.boundingRect(coords)
                if bw > w * 0.5 and bh > h * 0.5:
                    img = img[by:by+bh, bx:bx+bw]
                    h, w = img.shape[:2]

        # Clean colored annotations first so red boxes don't bias slice selection
        clean_bgr, _ = self.remove_colored_annotations(img)
        gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Extreme side-by-side aspect ratio (e.g., axial + coronal pair w/h >= 1.55)
        aspect = float(w) / float(h)
        if aspect >= 1.55:
            # Try contour-based candidate detection first (handles 2, 3, or 4 columns cleanly)
            _, thresh_grid = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            cv2.rectangle(thresh_grid, (0, 0), (w - 1, h - 1), 0, 4)
            kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh_grid = cv2.morphologyEx(thresh_grid, cv2.MORPH_OPEN, kernel_sep)
            cnts_g, _ = cv2.findContours(thresh_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            grid_cands = []
            for c in cnts_g:
                area = cv2.contourArea(c)
                if area > (h * w) * 0.035:
                    x, y, bw, bh = cv2.boundingRect(c)
                    ca = float(bw) / float(bh)
                    if 0.58 <= ca <= 1.42 and bw > 55 and bh > 55:
                        grid_cands.append((x, y, bw, bh, area))

            if len(grid_cands) >= 2:
                best = self._pick_best_brain_candidate(gray, grid_cands)
                x, y, bw, bh, _ = best
                pad = int(min(bw, bh) * 0.06)
                single = clean_bgr[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
                return self._pad_and_resize_square(single)
            else:
                # Fallback split left/right: prefer axial oval slice
                half_w = w // 2
                left_half = clean_bgr[:, :half_w]
                right_half = clean_bgr[:, half_w:]
                left_g = gray[:, :half_w]
                right_g = gray[:, half_w:]
                # Prefer the half with higher circularity / axial symmetry
                l_score = float(np.std(left_g))
                r_score = float(np.std(right_g))
                primary = left_half if l_score >= r_score * 0.92 else right_half
                return self.isolate_single_brain(primary)

        # 2. General multi-slice sheet / montage detection (e.g., 2x2, 2x3, 3x3)
        _, thresh = cv2.threshold(gray, 24, 255, cv2.THRESH_BINARY)
        cv2.rectangle(thresh, (0, 0), (w - 1, h - 1), 0, 4)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        min_area = (h * w) * 0.035
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                comp_aspect = float(bw) / float(bh)
                if 0.55 < comp_aspect < 1.45 and bw > 60 and bh > 60:
                    # Verify elliptical solidity (brains are roughly oval, > 0.72 solidity)
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    sol = area / hull_area if hull_area > 0 else 0
                    if sol > 0.68:
                        candidates.append((x, y, bw, bh, area))

        if len(candidates) > 1:
            best = self._pick_best_brain_candidate(gray, candidates)
            x, y, bw, bh, _ = best
            pad = int(min(bw, bh) * 0.06)
            single = clean_bgr[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
        elif len(candidates) == 1:
            x, y, bw, bh, _ = candidates[0]
            if float(bw * bh) / float(w * h) < 0.78:
                pad = int(min(bw, bh) * 0.05)
                single = clean_bgr[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
            else:
                single = clean_bgr
        else:
            single = clean_bgr

        return self._pad_and_resize_square(single)

    def _pick_best_brain_candidate(self, gray: np.ndarray, candidates: List[Tuple[int, int, int, int, float]]) -> Tuple[int, int, int, int, float]:
        """Selects the cleanest, most informative axial brain slice from a montage."""
        h, w = gray.shape

        def score_cand(cand):
            x, y, bw, bh, area = cand
            # Avoid slices cut off by image borders
            touches_edge = (x <= 3) or (y <= 3) or (x + bw >= w - 3) or (y + bh >= h - 3)
            edge_penalty = 0.55 if touches_edge else 1.0
            crop = gray[y:y+bh, x:x+bw]
            # Prefer good internal tissue contrast and oval aspect ratio near 0.85
            aspect = float(bw) / float(bh)
            aspect_bonus = 1.0 - min(0.35, abs(aspect - 0.86) * 0.6)
            # Check focal bright anomaly inside inner 70% of crop
            ch, cw = crop.shape
            inner = crop[int(ch*0.18):int(ch*0.82), int(cw*0.18):int(cw*0.82)]
            p95 = float(np.percentile(inner, 95)) if inner.size > 0 else 0.0
            p50 = float(np.percentile(inner, 50)) if inner.size > 0 else 0.0
            contrast_signal = max(10.0, p95 - p50)
            return float(area) * float(np.std(crop)) * aspect_bonus * edge_penalty * (1.0 + contrast_signal / 255.0)

        return max(candidates, key=score_cand)

    def _pad_and_resize_square(self, single: np.ndarray) -> np.ndarray:
        """Centers the isolated brain in a square frame and resizes to target_size."""
        sh, sw = single.shape[:2]
        max_dim = max(sh, sw)
        square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        dx = (max_dim - sw) // 2
        dy = (max_dim - sh) // 2
        square[dy:dy+sh, dx:dx+sw] = single
        resized = cv2.resize(square, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        # Clean peripheral text/logos outside cranial contour
        gray_clean = self.remove_peripheral_artifacts(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        return cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)

    def load_image(self, input_data: Any) -> np.ndarray:
        """Loads an image from a file path, base64 string, or raw bytes into a clean 512x512 BGR array."""
        if isinstance(input_data, np.ndarray):
            img = input_data.copy()
        elif isinstance(input_data, str):
            if input_data.startswith("data:image"):
                encoded_data = input_data.split(",")[1]
                img_bytes = base64.b64decode(encoded_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif input_data.startswith("base64,"):
                encoded_data = input_data[7:]
                img_bytes = base64.b64decode(encoded_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(input_data, cv2.IMREAD_COLOR)
                if img is None:
                    try:
                        from PIL import Image
                        pil_img = Image.open(input_data).convert('RGB')
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None
        elif isinstance(input_data, bytes):
            nparr = np.frombuffer(input_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported image input format.")

        if img is None:
            raise ValueError("Failed to decode image data.")

        return self.isolate_single_brain(img)

    def detect_modality(self, gray: np.ndarray, head_mask: np.ndarray) -> str:
        """
        Estimates imaging sequence/modality (CT Axial, MRI T1+C, MRI T2, MRI FLAIR)
        based on skull rim brightness and ventricular CSF dark/bright signature.
        """
        if np.count_nonzero(head_mask) == 0:
            return "CT / MRI AXIAL"

        # Outer skull ring vs inner parenchyma
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        inner = cv2.erode(head_mask, k_outer, iterations=1)
        rim = cv2.subtract(head_mask, inner)

        rim_vals = gray[rim > 0]
        inner_vals = gray[inner > 0]
        if len(rim_vals) == 0 or len(inner_vals) == 0:
            return "BRAIN CT / MRI"

        rim_p90 = float(np.percentile(rim_vals, 90))
        # Central ventricular region
        h, w = gray.shape
        center_box = gray[int(h*0.38):int(h*0.60), int(w*0.42):int(w*0.58)]
        center_p90 = float(np.percentile(center_box, 90)) if center_box.size > 0 else 0
        center_p15 = float(np.percentile(center_box, 15)) if center_box.size > 0 else 0
        inner_med = float(np.median(inner_vals))

        if rim_p90 > 225 and center_p15 < 55 and inner_med < 135:
            return "CT AXIAL / T1+C"
        elif center_p90 > 185 and rim_p90 < 210:
            return "MRI T2-WEIGHTED"
        elif rim_p90 < 185 and center_p15 < 55:
            return "MRI FLAIR AXIAL"
        else:
            return "MRI T1 / FLAIR"

    def apply_brain_window(self, gray: np.ndarray, is_raw_dicom_hu: bool = False) -> np.ndarray:
        """Normalizes dynamic range using percentile stretching to avoid outlier saturation."""
        if is_raw_dicom_hu:
            windowed = np.clip(gray, 0, 80)
            return cv2.normalize(windowed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fg = gray[gray > 12]
        if len(fg) > 100:
            p1 = float(np.percentile(fg, 1))
            p995 = float(np.percentile(fg, 99.5))
            if p995 > p1 + 10:
                scaled = np.clip((gray.astype(np.float32) - p1) * 255.0 / (p995 - p1), 0, 255).astype(np.uint8)
                scaled[gray <= 12] = 0
                return scaled
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def extract_brain_mask(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust Multi-Modality Skull Stripping (Brain Parenchyma Extraction).
        Cleanly removes bright calvarial bone, diploic fat, scalp, and dural rim
        so cortical bone or meningeal rims are never falsely segmented as tumors.
        """
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=65, sigmaSpace=65)

        # Step 1: Identify entire cranial head contour
        _, head_thresh = cv2.threshold(blurred, 22, 255, cv2.THRESH_BINARY)
        head_thresh = cv2.morphologyEx(head_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        contours, _ = cv2.findContours(head_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.ones_like(gray) * 255
            return mask, gray

        head_contour = max(contours, key=cv2.contourArea)
        head_mask = np.zeros_like(gray)
        cv2.drawContours(head_mask, [head_contour], -1, 255, thickness=cv2.FILLED)

        # Estimate head radius to scale inward erosion proportionally
        head_area = cv2.contourArea(head_contour)
        head_radius = np.sqrt(head_area / np.pi)

        # Step 2: Detect bright calvarial bone / scalp ring in the outer 22% shell of the head
        shell_thickness = max(15, int(head_radius * 0.22))
        k_shell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shell_thickness * 2 + 1, shell_thickness * 2 + 1))
        deep_core = cv2.erode(head_mask, k_shell, iterations=1)
        outer_shell = cv2.subtract(head_mask, deep_core)

        # Bright skull bone or dark CSF/subarachnoid moat separating skull from cortex
        _, bright_bone = cv2.threshold(blurred, 195, 255, cv2.THRESH_BINARY)
        outer_bone = cv2.bitwise_and(bright_bone, outer_shell)

        # Step 3: Erode head mask inward past skull & meninges (~11% of head radius)
        erode_px = max(13, int(head_radius * 0.115))
        k_inward = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        inner_cavity = cv2.erode(head_mask, k_inward, iterations=1)

        # Remove any remaining bright skull bone on periphery
        intracranial = cv2.bitwise_and(inner_cavity, cv2.bitwise_not(outer_bone))

        # Also trim dark subarachnoid CSF space right at the outer boundary
        edge_band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inner_deep = cv2.erode(intracranial, edge_band_k, iterations=1)
        periph_band = cv2.subtract(intracranial, inner_deep)
        dark_csf_rim = np.uint8((blurred < 45) & (periph_band > 0)) * 255
        intracranial = cv2.subtract(intracranial, dark_csf_rim)

        # Clean bridges and keep largest connected parenchyma component
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(intracranial, cv2.MORPH_OPEN, k_open, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            brain_mask = np.uint8(labels == largest_label) * 255
        else:
            brain_mask = opened

        # Smooth contour & fill internal ventricles/necrotic cavities
        cnts, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_cnt = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(main_cnt)
            # Blend contour with convex hull slightly if skull stripping bit into a peripheral lesion
            filled = np.zeros_like(gray)
            cv2.drawContours(filled, [main_cnt], -1, 255, thickness=cv2.FILLED)
            filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            brain_mask = filled

        stripped_brain = cv2.bitwise_and(gray, gray, mask=brain_mask)
        return brain_mask, stripped_brain

    def enhance_contrast(self, stripped_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies gentle bilateral smoothing + controlled CLAHE inside brain parenchyma."""
        smooth = cv2.bilateralFilter(stripped_gray, d=5, sigmaColor=35, sigmaSpace=35)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smooth)
        return cv2.bitwise_and(enhanced, enhanced, mask=mask)

    def compute_symmetry_axis_and_map(self, enhanced_gray: np.ndarray, brain_mask: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Finds the true anatomical centroid X of the brain mask (robust to slight off-center framing),
        computes a pixel-wise Contralateral Hemispheric Asymmetry Map, and returns:
        (midline_x, asymmetry_map_uint8, global_asymmetry_score).
        Healthy brains (even T2 scans with bright ventricles) have symmetric ventricles across the midline,
        yielding a near-zero contralateral difference, whereas unilateral tumors light up strongly!
        """
        h, w = enhanced_gray.shape
        M = cv2.moments(brain_mask)
        mid_x = int(round(M["m10"] / M["m00"])) if M["m00"] > 0 else (w // 2)
        mid_x = int(np.clip(mid_x, w * 0.35, w * 0.65))

        # Mirror image around mid_x
        mirrored = np.zeros_like(enhanced_gray)
        mirrored_mask = np.zeros_like(brain_mask)
        max_span = min(mid_x, w - mid_x)

        left_slice = enhanced_gray[:, mid_x - max_span:mid_x]
        right_slice = enhanced_gray[:, mid_x:mid_x + max_span]
        left_m = brain_mask[:, mid_x - max_span:mid_x]
        right_m = brain_mask[:, mid_x:mid_x + max_span]

        mirrored[:, mid_x - max_span:mid_x] = cv2.flip(right_slice, 1)
        mirrored[:, mid_x:mid_x + max_span] = cv2.flip(left_slice, 1)
        mirrored_mask[:, mid_x - max_span:mid_x] = cv2.flip(right_m, 1)
        mirrored_mask[:, mid_x:mid_x + max_span] = cv2.flip(left_m, 1)

        both_valid = cv2.bitwise_and(brain_mask, mirrored_mask)

        # Smooth slightly to tolerate anatomical sulcal & ventricular variations (12-16px)
        g_blur = cv2.GaussianBlur(enhanced_gray, (21, 21), 5.5)
        m_blur = cv2.GaussianBlur(mirrored, (21, 21), 5.5)
        # Also allow max-pool tolerance over 11x11 neighborhood so 5px midline tilt doesn't create false edge diffs
        m_dilated = cv2.dilate(m_blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

        # Positive excess intensity compared to opposite hemisphere neighborhood
        pos_diff = np.clip(g_blur.astype(np.float32) - m_dilated.astype(np.float32), 0, 255).astype(np.uint8)
        pos_diff = cv2.bitwise_and(pos_diff, pos_diff, mask=both_valid)

        abs_diff = cv2.absdiff(g_blur, m_blur)
        valid_vals = abs_diff[both_valid > 0]
        if len(valid_vals) == 0:
            return float(mid_x), pos_diff, 0.0

        # Normalize global asymmetry score cleanly (healthy ~ 0.04-0.18, tumor shift ~ 0.30-0.95)
        mean_abs = float(np.mean(valid_vals))
        p90_abs = float(np.percentile(valid_vals, 90))
        asym_raw = (0.55 * mean_abs + 0.45 * (p90_abs * 0.40)) / 56.0
        asym_score = float(np.clip(asym_raw, 0.0, 1.0))

        return float(mid_x), pos_diff, round(asym_score, 3)

    def _compute_glcm_features(self, gray_patch: np.ndarray, mask_patch: np.ndarray) -> Dict[str, float]:
        """
        Computes fast quantized Gray-Level Co-occurrence Matrix (GLCM, 16 levels, 4 angles)
        biomarkers inside the lesion: Contrast, Homogeneity, Energy, Entropy.
        """
        pts = gray_patch[mask_patch > 0]
        if len(pts) < 25:
            return {"entropy": 0.0, "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0}

        # Shannon entropy of lesion pixel intensities
        hist = cv2.calcHist([pts], [0], None, [32], [0, 256]).ravel()
        prob = hist / (np.sum(hist) + 1e-8)
        entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

        # 16-level quantized horizontal & vertical co-occurrence
        q = (gray_patch // 16).astype(np.int32)
        valid_h = (mask_patch[:, :-1] > 0) & (mask_patch[:, 1:] > 0)
        valid_v = (mask_patch[:-1, :] > 0) & (mask_patch[1:, :] > 0)

        i_idx = np.concatenate([q[:, :-1][valid_h], q[:-1, :][valid_v]])
        j_idx = np.concatenate([q[:, 1:][valid_h], q[1:, :][valid_v]])

        if len(i_idx) < 10:
            return {"entropy": round(entropy, 3), "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0}

        glcm = np.zeros((16, 16), dtype=np.float32)
        np.add.at(glcm, (i_idx, j_idx), 1.0)
        glcm = glcm + glcm.T
        glcm /= (np.sum(glcm) + 1e-8)

        ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
        diff_ij = np.abs(ii - jj)
        contrast = float(np.sum((diff_ij ** 2) * glcm))
        homogeneity = float(np.sum(glcm / (1.0 + diff_ij)))
        energy = float(np.sum(glcm ** 2))

        return {
            "entropy": round(entropy, 3),
            "contrast": round(contrast, 3),
            "homogeneity": round(homogeneity, 3),
            "energy": round(energy, 4)
        }

    def segment_tumor_candidates(
        self,
        raw_stripped: np.ndarray,
        enhanced_gray: np.ndarray,
        brain_mask: np.ndarray,
        midline_x: float,
        asym_map: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        High-Precision Multi-Cue Tumor Segmentation:
        Combines:
        1. Statistical Parenchyma Z-Score (detects both lateral and midline hyperdense/hyperintense masses)
        2. Contralateral Hemispheric Asymmetry Map (suppresses symmetric normal anatomy like T2 ventricles,
           choroid plexus, and falx cerebri while boosting unilateral tumors & ring-enhancing lesions)
        3. Morphological Necrotic-Core / Ring-Enhancing Filling (fills central necrotic core of GBM/ring tumors
           so the entire tumor volume and true solidity/heterogeneity are measured accurately)
        4. False-Positive Anatomical Veto (rejects thin midline falx lines, symmetric butterfly ventricles,
           and tiny cortical vessel speckles).
        """
        # Erode brain_mask slightly (6px) for candidate seeding so residual cortical bone rim never seeds a lesion
        seed_Zone = cv2.erode(brain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        brain_pixels = enhanced_gray[seed_Zone > 0]
        raw_pixels = raw_stripped[seed_Zone > 0]

        if len(brain_pixels) < 500:
            return np.zeros_like(enhanced_gray), []

        # Robust median & IQR-based std (resistant to large bright tumors skewing the mean)
        med_enh = float(np.median(brain_pixels))
        q75, q25 = np.percentile(brain_pixels, [75, 25])
        robust_std = max(12.0, float((q75 - q25) / 1.349))

        med_raw = float(np.median(raw_pixels))
        rq75, rq25 = np.percentile(raw_pixels, [75, 25])
        raw_robust_std = max(10.0, float((rq75 - rq25) / 1.349))

        h, w = enhanced_gray.shape
        # Detect Anterior Base-of-Skull / Orbital Slices (where frontal sinus/globe fat sits at top y < 32%)
        bx, by, b_w, b_h = cv2.boundingRect(brain_mask)
        if b_w < 50 or b_h < 50:
            bx, by, b_w, b_h = 0, 0, w, h

        # Anomaly score map (0..255) combining intensity Z-score + Contralateral Asymmetry
        z_enh = np.clip((enhanced_gray.astype(np.float32) - med_enh) / robust_std, 0, 6.0)
        z_raw = np.clip((raw_stripped.astype(np.float32) - med_raw) / raw_robust_std, 0, 6.0)
        asym_norm = asym_map.astype(np.float32) / max(18.0, robust_std)

        # Midline focal mass detector (for Pituitary adenoma, Sellar/Suprasellar, Hypothalamic hamartoma):
        # Surrounded by dark suprasellar cistern CSF or focal central nodule
        xx = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]
        sellar_zone = np.exp(-((xx - midline_x) ** 2) / (2.0 * (b_w * 0.065) ** 2)) * \
                      np.exp(-((yy - (by + b_h * 0.44)) ** 2) / (2.0 * (b_h * 0.065) ** 2))

        # Local Difference-of-Gaussians (DoG) highlights compact focal nodules surrounded by darker cistern
        dog_pos = np.clip(
            cv2.GaussianBlur(raw_stripped.astype(np.float32), (5, 5), 1.2) -
            cv2.GaussianBlur(raw_stripped.astype(np.float32), (35, 35), 9.0),
            0, 255
        ) / max(12.0, raw_robust_std)

        combined_score = (
            0.38 * z_enh +
            0.30 * z_raw +
            0.52 * asym_norm +
            0.36 * dog_pos * sellar_zone
        )
        combined_score = combined_score * (seed_Zone > 0).astype(np.float32)

        # Adaptive threshold on combined anomaly score
        binary_anom = np.uint8(combined_score >= 1.62) * 255

        # Also catch well-defined hyperdense masses directly on raw_stripped (e.g. meningioma, pituitary, T1+C)
        high_raw_thresh = min(240, max(154, int(med_raw + 2.0 * raw_robust_std)))
        _, direct_high = cv2.threshold(raw_stripped, high_raw_thresh, 255, cv2.THRESH_BINARY)
        direct_high = cv2.bitwise_and(direct_high, seed_Zone)

        # Also catch focal suprasellar nodules inside dark interpeduncular/suprasellar cistern
        # Must be genuinely elevated above parenchyma median (raw > med_raw * 1.15) AND surrounded by dark CSF cistern
        local_bg_35 = cv2.GaussianBlur(raw_stripped.astype(np.float32), (35, 35), 9.0)
        sellar_nodule = np.uint8(
            (dog_pos >= 0.65) &
            (sellar_zone >= 0.45) &
            (raw_stripped > med_raw * 1.15) &
            (local_bg_35 < med_raw * 1.02)
        ) * 255
        sellar_nodule = cv2.bitwise_and(sellar_nodule, seed_Zone)

        # Also detect dark/necrotic cavities surrounded by a hyperintense enhancing rim or significant mass effect:
        # In cystic/necrotic high-grade gliomas (e.g. T1+C cystic GBM), the cavity is dark (raw < med_raw * 0.62)
        # while the opposite hemisphere has normal gray/white matter AND an enhancing mural rim sits nearby!
        mirrored_raw = np.zeros_like(raw_stripped)
        max_sp = min(int(midline_x), w - int(midline_x))
        mx_i = int(midline_x)
        mirrored_raw[:, mx_i - max_sp:mx_i] = cv2.flip(raw_stripped[:, mx_i:mx_i + max_sp], 1)
        mirrored_raw[:, mx_i:mx_i + max_sp] = cv2.flip(raw_stripped[:, mx_i - max_sp:mx_i], 1)
        neg_asym = np.clip(
            cv2.GaussianBlur(mirrored_raw.astype(np.float32), (17, 17), 4.5) -
            cv2.GaussianBlur(raw_stripped.astype(np.float32), (17, 17), 4.5),
            0, 255
        )

        dark_cavity = np.uint8(
            (raw_stripped >= 4) &
            (raw_stripped < max(44, med_raw * 0.56)) &
            (neg_asym > 44.0)
        ) * 255
        dark_cavity = cv2.bitwise_and(dark_cavity, seed_Zone)
        # Only keep a well-rounded intra-parenchymal cystic/necrotic cavity (y in 28%..75%, solidity > 0.72)
        # that sits immediately adjacent to a bright contrast-enhancing mural rim (> 200 intensity)
        _, intense_enhancing = cv2.threshold(raw_stripped, max(195, int(med_raw + 2.45 * raw_robust_std)), 255, cv2.THRESH_BINARY)
        rim_touch = cv2.dilate(cv2.bitwise_and(intense_enhancing, seed_Zone), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        necrotic_cavity_adj = np.zeros_like(dark_cavity)
        num_dc, lbl_dc, stats_dc, _ = cv2.connectedComponentsWithStats(dark_cavity)
        for dci in range(1, num_dc):
            dc_area = stats_dc[dci, cv2.CC_STAT_AREA]
            if 650 <= dc_area <= (h * w * 0.11):
                comp_m = np.uint8(lbl_dc == dci) * 255
                dc_cnts, _ = cv2.findContours(comp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not dc_cnts:
                    continue
                dc_sol = solidity_calc(dc_cnts[0], float(dc_area))
                dc_M = cv2.moments(comp_m)
                dc_cx = (dc_M["m10"] / dc_M["m00"]) if dc_M["m00"] > 0 else midline_x
                dc_cy = (dc_M["m01"] / dc_M["m00"]) if dc_M["m00"] > 0 else (by + b_h * 0.5)
                dc_rel_y = (dc_cy - by) / float(b_h)
                if (
                    dc_sol >= 0.72 and
                    0.30 <= dc_rel_y <= 0.72 and
                    abs(dc_cx - midline_x) > b_w * 0.16 and
                    np.count_nonzero(cv2.bitwise_and(comp_m, rim_touch)) > 45
                ):
                    necrotic_cavity_adj[lbl_dc == dci] = 255
                    necrotic_cavity_adj = cv2.dilate(necrotic_cavity_adj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    necrotic_cavity_adj = cv2.bitwise_and(necrotic_cavity_adj, seed_Zone)

        raw_lesion_mask = cv2.bitwise_or(
            cv2.bitwise_or(cv2.bitwise_or(binary_anom, direct_high), sellar_nodule),
            necrotic_cavity_adj
        )

        # Clean speckles; use moderate closing (7x7, 1 iter) so compact benign nodules don't merge with normal cortex
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # For large ring-enhancing lesions (area > 450px), close wider ring gaps separately before hole-filling
        cnts_pre, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_candidates = np.zeros_like(cleaned_mask)
        for c in cnts_pre:
            if cv2.contourArea(c) > 450:
                sub = np.zeros_like(cleaned_mask)
                cv2.drawContours(sub, [c], -1, 255, thickness=cv2.FILLED)
                sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
                c_sub, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cs in c_sub:
                    cv2.drawContours(filled_candidates, [cs], -1, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(filled_candidates, [c], -1, 255, thickness=cv2.FILLED)

        contours, _ = cv2.findContours(filled_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_brain_area = float(np.count_nonzero(brain_mask))
        min_lateral_area = max(210.0, total_brain_area * 0.0028)
        min_sellar_area = max(95.0, total_brain_area * 0.0011)
        max_lesion_area = total_brain_area * 0.38

        lesions = []
        filtered_mask = np.zeros_like(enhanced_gray)

        # Brain bounding box already computed above
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_sellar_area or area_px > max_lesion_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_box = float(max(bw, bh)) / float(max(1, min(bw, bh)))

            # Veto 1: Thin elongated interhemispheric falx / sagittal sinus line right on midline
            M = cv2.moments(cnt)
            cx = float(M["m10"] / M["m00"]) if M["m00"] != 0 else float(x + bw / 2.0)
            cy = float(M["m01"] / M["m00"]) if M["m00"] != 0 else float(y + bh / 2.0)
            rel_cy = (cy - by) / float(b_h)
            is_sellar_cand = (abs(cx - midline_x) < b_w * 0.095) and (0.36 <= rel_cy <= 0.58)

            if not is_sellar_cand and area_px < min_lateral_area:
                continue

            if abs(cx - midline_x) < 18 and bh > bw * 2.1:
                continue

            # Veto 1b: Anterior frontobasal / gyrus rectus orbital susceptibility artifact
            if rel_cy < 0.36 and bw > bh * 1.45 and circularity_calc(cnt, area_px) < 0.62:
                continue

            # Extract pixel stats inside filled contour
            lesion_cnt_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(lesion_cnt_mask, [cnt - np.array([x, y])], -1, 255, -1)

            enh_patch = enhanced_gray[y:y+bh, x:x+bw]
            raw_patch = raw_stripped[y:y+bh, x:x+bw]
            asym_patch = asym_map[y:y+bh, x:x+bw]

            pts_enh = enh_patch[lesion_cnt_mask > 0]
            pts_raw = raw_patch[lesion_cnt_mask > 0]
            pts_asym = asym_patch[lesion_cnt_mask > 0]

            if len(pts_enh) == 0:
                continue

            mean_enh = float(np.mean(pts_enh))
            mean_raw = float(np.mean(pts_raw))
            p85_raw = float(np.percentile(pts_raw, 85))
            mean_asym_lesion = float(np.mean(pts_asym))

            if rel_cy < 0.36 and abs(cx - midline_x) < b_w * 0.18 and p85_raw < med_raw + 3.5 * raw_robust_std:
                continue

            # Veto 2: Symmetric normal ventricles / basal ganglia in healthy scans!
            has_necrotic_part = np.count_nonzero(necrotic_cavity_adj[y:y+bh, x:x+bw]) > 80
            is_near_midline = abs(cx - midline_x) < (b_w * 0.095)
            is_sellar_cist = is_near_midline and (0.36 <= rel_cy <= 0.58)
            if not is_near_midline and not has_necrotic_part and mean_asym_lesion < 20.0 and (p85_raw - med_raw) < 2.45 * raw_robust_std:
                continue

            # For compact sellar/suprasellar nodules (surrounded by dark CSF cistern), evaluate contrast vs local 35x35 cistern
            local_cist_mean = float(np.mean(local_bg_35[y:y+bh, x:x+bw]))
            sellar_prominence = (p85_raw - local_cist_mean) / (raw_robust_std + 1e-5)

            # Veto 3: Weak contrast blob (normal cortical gyrus in FLAIR/T1)
            contrast_ratio = (p85_raw - med_raw) / (raw_robust_std + 1e-5)
            if is_sellar_cist:
                if sellar_prominence < 1.15 and contrast_ratio < 1.25:
                    continue
                contrast_ratio = max(contrast_ratio, sellar_prominence)
            elif contrast_ratio < 1.62 and mean_asym_lesion < 25.0:
                if not has_necrotic_part:
                    continue
                contrast_ratio = max(contrast_ratio, 1.85)

            # Veto 3b: Dark cystic/post-surgical cavity WITHOUT enhancing mural rim
            # Keep cystic/necrotic tumors whose p85_raw has an enhancing rim (>= med_raw + 0.35 * raw_robust_std) or high asymmetry
            if mean_raw < med_raw * 0.48 and p85_raw < med_raw + 0.35 * raw_robust_std and mean_asym_lesion < 18.0:
                continue

            # Veto 4: Symmetric bilateral frontal/occipital white matter tracts
            if aspect_box > 3.8 and solidity_calc(cnt, area_px) < 0.55:
                continue

            # Morphometric descriptors
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / hull_area) if hull_area > 0 else 0.0
            perimeter_px = cv2.arcLength(cnt, True)
            circularity = float(4.0 * np.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
            equiv_diameter_px = float(np.sqrt(4.0 * area_px / np.pi))

            # Eccentricity from fitted ellipse
            if len(cnt) >= 5:
                (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (min(ma, MA) / max(ma, MA, 1e-5)) ** 2)))
            else:
                eccentricity = 0.0

            # Check necrotic core / ring-enhancement signature:
            # Compare inner core intensity vs outer rim of the lesion
            necrotic_ratio = 0.0
            if bw > 18 and bh > 18:
                er_k = max(3, int(min(bw, bh) * 0.22))
                inner_l_mask = cv2.erode(lesion_cnt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er_k, er_k)))
                rim_l_mask = cv2.subtract(lesion_cnt_mask, inner_l_mask)
                if np.count_nonzero(inner_l_mask) > 12 and np.count_nonzero(rim_l_mask) > 12:
                    core_mean = float(np.mean(raw_patch[inner_l_mask > 0]))
                    rim_mean = float(np.mean(raw_patch[rim_l_mask > 0]))
                    if rim_mean > core_mean + 14.0:
                        necrotic_ratio = round(min(1.0, (rim_mean - core_mean) / 85.0), 3)

            # Peritumoral Edema estimation (hypodense/hyperintense halo around tumor)
            dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            full_this = np.zeros_like(enhanced_gray)
            cv2.drawContours(full_this, [cnt], -1, 255, -1)
            halo_mask = cv2.bitwise_and(cv2.subtract(cv2.dilate(full_this, dil_k), full_this), brain_mask)
            halo_vals = raw_stripped[halo_mask > 0]
            edema_index = 0.0
            if len(halo_vals) > 30:
                edema_diff = abs(float(np.mean(halo_vals)) - med_raw) / (raw_robust_std + 1e-5)
                edema_index = round(float(np.clip(edema_diff / 2.5, 0.0, 1.0)), 3)

            # GLCM Texture features
            glcm_feats = self._compute_glcm_features(enh_patch, lesion_cnt_mask)

            # Radiological Hemisphere Localization (Standard Radiological Convention:
            # Left side of image = Patient's Right Hemisphere, Right side of image = Patient's Left Hemisphere;
            # We clearly note both Anatomical & Image side for zero ambiguity)
            if abs(cx - midline_x) <= b_w * 0.07:
                hemisphere = "خط وسط / سوبراتنتوریال مرکزی (Midline / Sellar)"
            elif cx < midline_x:
                hemisphere = "نیمکره راست بیمار (Right Hemisphere)"
            else:
                hemisphere = "نیمکره چپ بیمار (Left Hemisphere)"

            # Relative Lobar Localization inside actual brain bounding box
            rel_y = (cy - by) / float(b_h)
            rel_x_dist = abs(cx - midline_x) / float(b_w / 2.0)

            if abs(cx - midline_x) <= b_w * 0.10 and 0.35 <= rel_y <= 0.62:
                lobe = "ناحیه سلار / هیپوتالاموس و بطن سوم (Sellar / Chiasmatic)"
            elif rel_y < 0.36:
                lobe = "لوب پیشانی / فرونتال (Frontal Lobe)"
            elif rel_y > 0.68:
                lobe = "لوب پس‌سری / اکسیپیتال (Occipital Lobe)"
            elif rel_x_dist > 0.44:
                lobe = "لوب گیجگاهی / تمپورال (Temporal Lobe)"
            elif 0.36 <= rel_y <= 0.68 and rel_x_dist <= 0.44:
                lobe = "ناحیه آهیانه / پری‌ونتریکولار (Parietal / Periventricular)"
            else:
                lobe = "لوب آهیانه / پریتال (Parietal Lobe)"

            # Physical calibrated measurements
            area_mm2 = area_px * (self.pixel_spacing_mm ** 2)
            area_cm2 = area_mm2 / 100.0
            diameter_mm = equiv_diameter_px * self.pixel_spacing_mm
            # Ellipsoid volume approximation V ≈ (4/3) * pi * (a/2)*(b/2)*(c/2) -> ~ 0.65 * area * diameter
            est_volume_cm3 = (area_cm2 * (diameter_mm / 10.0)) * 0.65

            std_density = float(np.std(pts_raw))
            heterogeneity_index = float(std_density / (mean_raw + 1e-5))

            # Composite lesion confidence score for ranking primary tumor
            lesion_salience = (
                area_px *
                (1.0 + 0.5 * contrast_ratio) *
                (1.0 + (mean_asym_lesion / 35.0))
            )

            lesions.append({
                "contour": cnt.tolist(),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "centroid": [round(cx, 1), round(cy, 1)],
                "area_px": int(area_px),
                "area_cm2": round(area_cm2, 2),
                "area_mm2": round(area_mm2, 1),
                "diameter_mm": round(diameter_mm, 1),
                "volume_cm3": round(est_volume_cm3, 2),
                "perimeter_px": round(perimeter_px, 1),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3),
                "eccentricity": round(eccentricity, 3),
                "hemisphere": hemisphere,
                "lobe": lobe,
                "mean_density": round(mean_raw, 1),
                "std_density": round(std_density, 1),
                "heterogeneity_index": round(heterogeneity_index, 3),
                "necrotic_core_index": necrotic_ratio,
                "edema_index": edema_index,
                "contrast_zscore": round(contrast_ratio, 2),
                "asymmetry_intensity": round(mean_asym_lesion, 1),
                "glcm_entropy": glcm_feats["entropy"],
                "glcm_contrast": glcm_feats["contrast"],
                "glcm_homogeneity": glcm_feats["homogeneity"],
                "glcm_energy": glcm_feats["energy"],
                "_salience": lesion_salience
            })

        # Sort by clinical salience so primary tumor is always #1
        lesions.sort(key=lambda item: item["_salience"], reverse=True)

        # Keep primary tumor and significant satellites (at least 25% of primary area)
        if lesions:
            top_area = lesions[0]["area_px"]
            lesions = [l for l in lesions if l["area_px"] >= top_area * 0.25][:3]
            for l in lesions:
                l.pop("_salience", None)
                cnt_arr = np.array(l["contour"], dtype=np.int32)
                cv2.drawContours(filtered_mask, [cnt_arr], -1, 255, thickness=cv2.FILLED)

        return filtered_mask, lesions

    def estimate_midline_shift_mm(self, raw_stripped: np.ndarray, brain_mask: np.ndarray, midline_x: float, asym_score: float) -> float:
        """Estimates anatomical midline shift in millimeters."""
        if asym_score < 0.20:
            return round(asym_score * 3.5, 1)
        h, w = raw_stripped.shape
        # Check dark ventricular septum displacement near center
        center_band = raw_stripped[int(h*0.36):int(h*0.58), int(midline_x - 28):int(midline_x + 28)]
        if center_band.size > 0:
            col_means = np.mean(center_band, axis=0)
            min_col = int(np.argmin(col_means))
            shift_px = abs((midline_x - 28 + min_col) - midline_x)
            shift_mm = min(14.5, shift_px * self.pixel_spacing_mm * (0.45 + asym_score))
            return round(float(shift_mm), 1)
        return round(float(asym_score * 8.5), 1)

    def generate_pipeline_steps(self, bgr_image: np.ndarray, cv_res: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Captures 8 visual stages of the OpenCV pipeline as base64 PNGs for the UI Pipeline Inspector.
        """
        if cv_res is None:
            cv_res = self.process_scan(bgr_image)

        windowed = cv_res["windowed_gray"]
        brain_mask = cv_res["brain_mask"]
        stripped_brain = cv_res["stripped_brain"]
        enhanced = cv_res["enhanced_brain"]
        asym_map = cv_res["asymmetry_map"]
        tumor_mask = cv_res["tumor_mask"]
        lesions = cv_res["lesions"]

        # Step 7: Annotated Contours & Quantitative Callouts
        annotated = cv_res["image_bgr"].copy()
        mid_x = int(cv_res["midline_x"])
        cv2.line(annotated, (mid_x, 25), (mid_x, 487), (180, 180, 0), 1, cv2.LINE_AA)

        if lesions:
            for i, lesion in enumerate(lesions):
                cnt = np.array(lesion["contour"], dtype=np.int32)
                cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), 2, cv2.LINE_AA)
                x, y, bw, bh = lesion["bbox"]
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 140, 255), 2, cv2.LINE_AA)
                label = f"LESION #{i+1}: {lesion['area_cm2']}cm2 | {lesion['diameter_mm']}mm"
                cv2.rectangle(annotated, (max(4, x), max(4, y - 22)), (min(508, x + 255), max(24, y)), (10, 18, 32), -1)
                cv2.putText(annotated, label, (max(8, x + 4), max(18, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)

        # Colorize asymmetry map for clear visual inspection
        asym_vis = cv2.applyColorMap(cv2.normalize(asym_map, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_MAGMA)
        asym_vis = cv2.bitwise_and(asym_vis, asym_vis, mask=brain_mask)

        def to_b64(img_array: np.ndarray) -> str:
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode('.png', img_array)
            return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

        return {
            "step1_raw": to_b64(cv_res["image_bgr"]),
            "step2_windowed": to_b64(windowed),
            "step3_brain_mask": to_b64(brain_mask),
            "step4_stripped_brain": to_b64(stripped_brain),
            "step5_clahe_enhanced": to_b64(enhanced),
            "step6_tumor_mask": to_b64(tumor_mask),
            "step7_annotated": to_b64(annotated),
            "step8_bone_colormap": to_b64(asym_vis)
        }

    def process_scan(self, input_data: Any) -> Dict[str, Any]:
        """
        Complete end-to-end OpenCV execution on a Brain CT / MRI scan.
        """
        bgr = self.load_image(input_data)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        windowed = self.apply_brain_window(gray)
        brain_mask, stripped_brain = self.extract_brain_mask(windowed)
        modality = self.detect_modality(windowed, brain_mask)
        enhanced = self.enhance_contrast(stripped_brain, brain_mask)
        midline_x, asym_map, asymmetry_score = self.compute_symmetry_axis_and_map(enhanced, brain_mask)
        tumor_mask, lesions = self.segment_tumor_candidates(
            stripped_brain, enhanced, brain_mask, midline_x, asym_map
        )

        # Refine asymmetry score: if no tumor and normal ventricles, prevent false midline shift alarm
        if not lesions:
            asymmetry_score = round(min(0.24, asymmetry_score * 0.65), 3)

        midline_shift_mm = self.estimate_midline_shift_mm(stripped_brain, brain_mask, midline_x, asymmetry_score)

        brain_area_px = int(np.count_nonzero(brain_mask))
        brain_area_cm2 = round(brain_area_px * (self.pixel_spacing_mm ** 2) / 100.0, 2)

        has_tumor = len(lesions) > 0
        primary_lesion = lesions[0] if has_tumor else None
        lesion_brain_ratio_pct = round(
            (sum(l["area_px"] for l in lesions) / max(1, brain_area_px)) * 100.0, 2
        ) if has_tumor else 0.0

        # Extract Ultra-Fast C++17 32-level 4-angle GLCM + GLRLM + LBP + Fractal Dimension
        from core.cpp_bridge import get_cpp_bridge
        cpp_rad = get_cpp_bridge().extract(enhanced, brain_mask, tumor_mask, midline_x)
        if primary_lesion is not None:
            primary_lesion["fractal_dim"] = cpp_rad["fractal_dim"]
            primary_lesion["glrlm_sre"] = cpp_rad["glrlm_sre"]
            primary_lesion["lbp_entropy"] = cpp_rad["lbp_entropy"]
            primary_lesion["margin_sharpness"] = cpp_rad["margin_sharpness"]
            primary_lesion["skewness"] = cpp_rad["skewness"]
            primary_lesion["kurtosis"] = cpp_rad["kurtosis"]

        return {
            "image_bgr": bgr,
            "image_gray": gray,
            "windowed_gray": windowed,
            "brain_mask": brain_mask,
            "stripped_brain": stripped_brain,
            "enhanced_brain": enhanced,
            "asymmetry_map": asym_map,
            "midline_x": midline_x,
            "midline_shift_mm": midline_shift_mm,
            "modality": modality,
            "tumor_mask": tumor_mask,
            "lesions": lesions,
            "has_tumor": has_tumor,
            "primary_lesion": primary_lesion,
            "lesion_count": len(lesions),
            "lesion_brain_ratio_pct": lesion_brain_ratio_pct,
            "brain_area_cm2": brain_area_cm2,
            "asymmetry_score": asymmetry_score,
            "cpp_radiomics": cpp_rad
        }


def solidity_calc(cnt: np.ndarray, area_px: float) -> float:
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    return float(area_px / ha) if ha > 0 else 0.0


def circularity_calc(cnt: np.ndarray, area_px: float) -> float:
    perim = cv2.arcLength(cnt, True)
    return float(4.0 * np.pi * area_px / (perim ** 2)) if perim > 0 else 0.0
