/**
 * NeuroScan AI - TypeScript Type Definitions & Domain Models
 * Comprehensive type definitions for brain CT/MRI computer vision,
 * radiomic biomarkers, AI diagnostics, and clinical oncology classifications.
 */

export type ScanModality = 'CT_AXIAL' | 'MRI_T1' | 'MRI_T2' | 'MRI_FLAIR' | 'CT_CONTRAST';

export type DiagnosisClass = 'normal' | 'benign' | 'malignant';

export type WHOGrade = 
  | 'Grade 0 (فاقد هرگونه ضایعه نئوپلاستیک)'
  | 'Grade I (تومور خوش‌خیم با رشد آهسته و حدود مشخص)'
  | 'Grade II (تومور با درجه پایین اما مستعد عود)'
  | 'Grade III (گلیوما آناپلاستیک بدخیم / High-Grade Glioma)'
  | 'Grade IV (بدخیم با تهاجم شدید - پترن Glioblastoma Multiforme)';

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LesionBiomarkers {
  contour: [number, number][];
  bbox: [number, number, number, number];
  centroid: [number, number];
  area_px: number;
  area_cm2: number;
  area_mm2: number;
  diameter_mm: number;
  perimeter_px: number;
  circularity: number;
  solidity: number;
  hemisphere: string;
  lobe: string;
  mean_density: number;
  std_density: number;
  heterogeneity_index: number;
}

export interface OpenCVMetrics {
  has_tumor: boolean;
  lesion_count: number;
  brain_area_cm2: number;
  asymmetry_score: number;
  primary_lesion: LesionBiomarkers | null;
  all_lesions: LesionBiomarkers[];
}

export interface MalignancyFactor {
  factor: string;
  impact: number;
  status: 'نرمال' | 'متوسط' | 'بالا';
}

export interface SubtypeBreakdownItem {
  name_fa: string;
  name_en: string;
  percent: number;
}

export interface AIDiagnosticReport {
  developer?: string;
  engine_version?: string;
  modality_detected?: string;
  diagnosis_id: DiagnosisClass;
  diagnosis_fa: string;
  diagnosis_en: string;
  diagnostic_confidence_pct?: number;
  ai_reliability_pct?: number;
  tta_consistency_pct?: number;
  segmentation_confidence_pct?: number;
  cancer_probability: number;
  cancer_probability_fraction: number;
  probabilities: {
    cancer_malignant: number;
    benign: number;
    normal: number;
  };
  subtype_breakdown?: SubtypeBreakdownItem[];
  who_grade: string;
  risk_level: RiskLevel;
  risk_label_fa: string;
  subpathology: string;
  clinical_recommendation: string;
  malignancy_factors: MalignancyFactor[];
  midline_shift_mm?: number;
  gradcam_heatmap_b64: string;
  diagnostic_overlay_b64: string;
}

export interface PipelineSteps {
  step1_raw: string;
  step2_windowed: string;
  step3_brain_mask: string;
  step4_stripped_brain: string;
  step5_clahe_enhanced: string;
  step6_tumor_mask: string;
  step7_annotated: string;
  step8_bone_colormap: string;
}

export interface AnalysisResponse {
  status: 'success' | 'error';
  opencv_metrics: OpenCVMetrics;
  ai_diagnosis: AIDiagnosticReport;
  pipeline_steps: PipelineSteps;
  message?: string;
}

export interface SampleCase {
  id: string;
  filename: string;
  title_fa: string;
  title_en: string;
  diagnosis: string;
  thumbnail_b64: string;
}

export interface CaliperMeasurement {
  start: { x: number; y: number };
  end: { x: number; y: number };
  distancePx: number;
  distanceMm: number;
}

export type ActiveLayer = 'overlay' | 'gradcam' | 'contours' | 'stripped' | 'raw';

export interface ViewerState {
  currentScanB64: string | null;
  analysisData: AnalysisResponse | null;
  pipelineSteps: PipelineSteps | null;
  activeLayer: ActiveLayer;
  heatmapOpacity: number;
  caliperActive: boolean;
  scannerActive: boolean;
  pixelSpacingMm: number;
}


// ===========================================================================
// NeuroScan AI Ultra v4.0 — C++17 Native Engine interfaces
// برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
// ===========================================================================

/** C++17 native radiomic biomarker bundle (4-angle GLCM / GLRLM / LBP / Fractal) */
export interface CppRadiomics {
  cpp_accelerated: boolean;
  glcm_entropy: number;
  glcm_contrast: number;
  glcm_homogeneity: number;
  glcm_energy: number;
  glrlm_sre: number;
  glrlm_lre: number;
  glrlm_rln: number;
  lbp_uniformity: number;
  lbp_entropy: number;
  fractal_dim: number;
  margin_sharpness: number;
  skewness: number;
  kurtosis: number;
  contralateral_corr: number;
}

/** Engine capability descriptor returned by /api/analyze */
export interface EngineInfo {
  name: string;
  native_cpp_engine: boolean;
  cpp_radiomics_active: boolean;
  neural_network: string;
  radiomic_features: string;
  calibrated_confidence_floor_pct: number;
  developer: string;
  developer_en: string;
}

/** Trained neural network 3-class probability vector */
export interface NeuralProbabilities {
  malignant: number;
  benign: number;
  normal: number;
}
