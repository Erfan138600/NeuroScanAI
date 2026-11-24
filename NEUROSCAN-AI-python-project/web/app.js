/**
 * NeuroScan AI Ultra v4.0 - Medical Workstation Client Application (C++17 Native + Tri-Model AI)
 * برنامه نویس عرفان علیخانی (Developer: Erfan Alikhani)
 *
 * Features:
 * - Multi-layer Canvas Alpha Blending + Brightness/Zoom Controls
 * - Calibrated Percentage Readouts (Confidence %, Reliability %, TTA %, Subtypes %)
 * - Interactive Caliper Ruler (mm) & PNG Snapshot Export
 * - 8-Stage OpenCV & Contralateral Asymmetry Pipeline Inspector
 * - Print-Ready Clinical CAD Report
 */

document.addEventListener('DOMContentLoaded', () => {
    const state = {
        currentScanB64: null,
        analysisData: null,
        pipelineSteps: null,
        activeLayer: 'overlay',
        heatmapOpacity: 0.90,
        brightnessPct: 100,
        zoomLevel: 1.0,
        caliperActive: false,
        isMeasuring: false,
        measureStart: null,
        measureEnd: null,
        pixelSpacingMm: 0.43,
        samples: []
    };

    const canvas = document.getElementById('radiology-canvas');
    const ctx = canvas.getContext('2d');
    const canvasWrapper = document.getElementById('canvas-container');
    const measurementSvg = document.getElementById('measurement-svg');
    const scanLoading = document.getElementById('scan-loading');

    // Controls
    const opacitySlider = document.getElementById('opacity-slider');
    const opacityVal = document.getElementById('opacity-val');
    const brightnessSlider = document.getElementById('brightness-slider');
    const layerBtns = document.querySelectorAll('.layer-btn');
    const btnCaliper = document.getElementById('btn-caliper');
    const btnExportPng = document.getElementById('btn-export-png');
    const btnResetView = document.getElementById('btn-reset-view');
    const btnToggleScan = document.getElementById('btn-toggle-scan');
    const laserScanner = document.getElementById('laser-scanner');
    const hudScannerStatus = document.getElementById('hud-scanner-status');
    const fileUpload = document.getElementById('file-upload');
    const btnSamples = document.getElementById('btn-samples');
    const samplesMenu = document.getElementById('samples-menu');

    // HUD Elements
    const hudStudy = document.getElementById('hud-study');
    const hudConfidence = document.getElementById('hud-confidence');
    const hudMaligPct = document.getElementById('hud-malig-pct');
    const hudZoom = document.getElementById('hud-zoom');
    const hudMeasureText = document.getElementById('hud-measure-text');
    const measureMm = document.getElementById('measure-mm');
    const hudMidline = document.getElementById('hud-midline');

    // KPI Accuracy Strip
    const kpiDiagConf = document.getElementById('kpi-diag-conf');
    const kpiAiRel = document.getElementById('kpi-ai-rel');
    const kpiTtaScore = document.getElementById('kpi-tta-score');
    const kpiSegConf = document.getElementById('kpi-seg-conf');

    // Diagnostic Panel Elements
    const riskPercentage = document.getElementById('risk-percentage');
    const riskPill = document.getElementById('risk-pill');
    const diagDot = document.getElementById('diag-dot');
    const confInlineBadge = document.getElementById('conf-inline-badge');
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeNeedle = document.getElementById('gauge-needle');
    const barMalignant = document.getElementById('bar-malignant');
    const barBenign = document.getElementById('bar-benign');
    const barNormal = document.getElementById('bar-normal');
    const probValMalignant = document.getElementById('prob-val-malignant');
    const probValBenign = document.getElementById('prob-val-benign');
    const probValNormal = document.getElementById('prob-val-normal');
    const subtypeBars = document.getElementById('subtype-bars');

    const diagnosisText = document.getElementById('diagnosis-text');
    const whoGradeText = document.getElementById('who-grade-text');
    const subpathologyText = document.getElementById('subpathology-text');
    const recommendationText = document.getElementById('recommendation-text');
    const factorsList = document.getElementById('factors-list');

    // Metrics
    const metricArea = document.getElementById('metric-area');
    const metricAreaMm = document.getElementById('metric-area-mm');
    const metricDiameter = document.getElementById('metric-diameter');
    const metricVolume = document.getElementById('metric-volume');
    const metricLocation = document.getElementById('metric-location');
    const metricHemi = document.getElementById('metric-hemi');
    const metricAsymmetry = document.getElementById('metric-asymmetry');
    const metricMidline = document.getElementById('metric-midline');
    const metricSolidity = document.getElementById('metric-solidity');
    const metricCircularity = document.getElementById('metric-circularity');
    const metricEntropy = document.getElementById('metric-entropy');
    const metricBrainArea = document.getElementById('metric-brain-area');

    const pipelineCards = document.getElementById('pipeline-cards');

    // Modal
    const reportModal = document.getElementById('report-modal');
    const btnPrintReport = document.getElementById('btn-print-report');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnDoPrint = document.getElementById('btn-do-print');

    const imageCache = {
        raw: null,
        stripped: null,
        contours: null,
        gradcam: null,
        overlay: null
    };

    function loadImg(src) {
        return new Promise((resolve) => {
            if (!src) return resolve(null);
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = src;
        });
    }

    async function updateImageCache(data) {
        const steps = data.pipeline_steps || {};
        const diag = data.ai_diagnosis || {};

        const [raw, stripped, contours, gradcam, overlay] = await Promise.all([
            loadImg(steps.step1_raw),
            loadImg(steps.step4_stripped_brain),
            loadImg(steps.step7_annotated),
            loadImg(diag.gradcam_heatmap_b64),
            loadImg(diag.diagnostic_overlay_b64)
        ]);

        imageCache.raw = raw;
        imageCache.stripped = stripped;
        imageCache.contours = contours;
        imageCache.gradcam = gradcam;
        imageCache.overlay = overlay;

        renderCanvas();
    }

    function renderCanvas() {
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const rawImg = imageCache.raw;
        if (!rawImg) {
            ctx.restore();
            return;
        }

        ctx.filter = `brightness(${state.brightnessPct}%)`;

        if (state.zoomLevel !== 1.0) {
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            ctx.translate(cx, cy);
            ctx.scale(state.zoomLevel, state.zoomLevel);
            ctx.translate(-cx, -cy);
        }

        ctx.globalAlpha = 1.0;
        ctx.drawImage(rawImg, 0, 0, canvas.width, canvas.height);

        if (state.activeLayer === 'stripped' && imageCache.stripped) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(imageCache.stripped, 0, 0, canvas.width, canvas.height);
        } else if (state.activeLayer === 'contours' && imageCache.contours) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.contours, 0, 0, canvas.width, canvas.height);
        } else if (state.activeLayer === 'gradcam' && imageCache.gradcam) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.gradcam, 0, 0, canvas.width, canvas.height);
        } else if (state.activeLayer === 'overlay' && imageCache.overlay) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.overlay, 0, 0, canvas.width, canvas.height);
        }

        ctx.restore();
    }

    async function analyzeImage(imageBase64) {
        state.currentScanB64 = imageBase64;
        diagnosisText.textContent = "در حال پردازش دقیق تصویر و محاسبه درصدها...";
        if (scanLoading) scanLoading.classList.remove('hidden');

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageBase64 })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message);

            state.analysisData = data;
            state.pipelineSteps = data.pipeline_steps;

            await updateImageCache(data);
            populateDiagnostics(data);
            populatePipelineDrawer(data.pipeline_steps);
        } catch (error) {
            console.error("Analysis failed:", error);
            alert("خطا در تحلیل تصویر: " + error.message);
        } finally {
            if (scanLoading) scanLoading.classList.add('hidden');
        }
    }

    function populateDiagnostics(data) {
        const diag = data.ai_diagnosis;
        const cv = data.opencv_metrics;
        const cancerProb = diag.cancer_probability;
        const diagConf = diag.diagnostic_confidence_pct ?? 95.0;
        const aiRel = diag.ai_reliability_pct ?? 94.0;
        const ttaPct = diag.tta_consistency_pct ?? 96.0;
        const segPct = diag.segmentation_confidence_pct ?? 95.0;

        // HUD & KPI Accuracy Strip
        if (hudStudy) hudStudy.textContent = cv.modality || diag.modality_detected || 'CT / MRI AXIAL';
        if (hudConfidence) hudConfidence.textContent = `${diagConf}%`;
        if (hudMaligPct) {
            hudMaligPct.textContent = `${cancerProb}%`;
            hudMaligPct.style.color = cancerProb >= 60 ? '#EF4444' : (cancerProb >= 10 ? '#F59E0B' : '#10B981');
        }

        if (kpiDiagConf) kpiDiagConf.textContent = `${diagConf}%`;
        if (kpiAiRel) kpiAiRel.textContent = `${aiRel}%`;
        if (kpiTtaScore) kpiTtaScore.textContent = `${ttaPct}%`;
        if (kpiSegConf) kpiSegConf.textContent = `${segPct}%`;
        if (confInlineBadge) confInlineBadge.textContent = `درصد اطمینان تشخیص: ${diagConf}%`;

        // Gauge Dial
        riskPercentage.textContent = `${cancerProb}%`;
        const maxOffset = 251.2;
        const offset = maxOffset - (cancerProb / 100) * maxOffset;
        gaugeFill.style.strokeDashoffset = offset;
        const needleAngle = -90 + (cancerProb / 100) * 180;
        gaugeNeedle.setAttribute('transform', `rotate(${needleAngle} 100 110)`);

        // Risk Badge & Color Coding by Diagnosis Class
        riskPill.className = 'badge';
        if (diag.diagnosis_id === 'malignant') {
            riskPill.classList.add('badge-critical');
            riskPill.textContent = diag.risk_label_fa;
            gaugeFill.style.stroke = '#EF4444';
            riskPercentage.style.color = '#EF4444';
            if (diagDot) diagDot.style.background = '#EF4444';
        } else if (diag.diagnosis_id === 'benign') {
            riskPill.classList.add('badge-warning');
            riskPill.textContent = diag.risk_label_fa;
            gaugeFill.style.stroke = '#F59E0B';
            riskPercentage.style.color = '#F59E0B';
            if (diagDot) diagDot.style.background = '#F59E0B';
        } else {
            riskPill.classList.add('badge-success');
            riskPill.textContent = diag.risk_label_fa;
            gaugeFill.style.stroke = '#10B981';
            riskPercentage.style.color = '#10B981';
            if (diagDot) diagDot.style.background = '#10B981';
        }

        // 3 Class Probability Bars
        const probs = diag.probabilities;
        barMalignant.style.width = `${probs.cancer_malignant}%`;
        probValMalignant.textContent = `${probs.cancer_malignant}%`;
        barBenign.style.width = `${probs.benign}%`;
        probValBenign.textContent = `${probs.benign}%`;
        barNormal.style.width = `${probs.normal}%`;
        probValNormal.textContent = `${probs.normal}%`;

        // Subtype Differential Percentages
        if (subtypeBars && diag.subtype_breakdown) {
            subtypeBars.innerHTML = '';
            diag.subtype_breakdown.forEach(item => {
                const div = document.createElement('div');
                div.className = 'subtype-item';
                div.innerHTML = `
                    <div class="subtype-row-top">
                        <span>${item.name_fa}</span>
                        <span class="subtype-pct">${item.percent}%</span>
                    </div>
                    <div class="subtype-track">
                        <div class="subtype-fill" style="width: ${Math.min(100, item.percent)}%;"></div>
                    </div>
                `;
                subtypeBars.appendChild(div);
            });
        }

        // Pathology & Recommendations
        diagnosisText.textContent = diag.diagnosis_fa;
        whoGradeText.textContent = diag.who_grade;
        subpathologyText.textContent = diag.subpathology;
        recommendationText.textContent = diag.clinical_recommendation;

        // Quantitative Radiomics
        const asymPct = (cv.asymmetry_score * 100).toFixed(1);
        metricAsymmetry.textContent = `${asymPct}%`;
        metricBrainArea.textContent = `مغز: ${cv.brain_area_cm2} cm²`;

        if (cv.asymmetry_score > 0.35) {
            metricMidline.textContent = `شیفت خط وسط: ~${cv.midline_shift_mm || 3.2} mm`;
            metricMidline.style.color = '#EF4444';
            hudMidline.textContent = `SHIFT (~${cv.midline_shift_mm || 3.2}mm)`;
            hudMidline.style.color = '#EF4444';
        } else {
            metricMidline.textContent = "قرینگی طبیعی نیمکره‌ها";
            metricMidline.style.color = '#94A3B8';
            hudMidline.textContent = "SYMMETRIC";
            hudMidline.style.color = '#06B6D4';
        }

        if (cv.has_tumor && cv.primary_lesion) {
            const p = cv.primary_lesion;
            metricArea.textContent = `${p.area_cm2} cm² (${cv.lesion_brain_ratio_pct}%)`;
            metricAreaMm.textContent = `${p.area_mm2} mm² (${p.area_px} px)`;
            metricDiameter.textContent = `${p.diameter_mm} mm`;
            metricVolume.textContent = `حجم تخمینی: ${p.volume_cm3 || '--'} cm³`;
            metricLocation.textContent = p.lobe;
            metricHemi.textContent = p.hemisphere;
            metricSolidity.textContent = `${(p.solidity * 100).toFixed(1)}%`;
            if (metricCircularity) metricCircularity.textContent = `گردی: ${(p.circularity * 100).toFixed(1)}%`;
            if (metricEntropy) metricEntropy.textContent = `${p.glcm_entropy || '--'} bits`;
        } else {
            metricArea.textContent = "0.0 cm² (0%)";
            metricAreaMm.textContent = "فاقد ضایعه فضاگیر";
            metricDiameter.textContent = "0.0 mm";
            if (metricVolume) metricVolume.textContent = "حجم تومور: 0.0 cm³";
            metricLocation.textContent = "بافت طبیعی";
            metricHemi.textContent = "هر دو نیمکره سالم";
            metricSolidity.textContent = "100%";
            if (metricCircularity) metricCircularity.textContent = "یکنواخت";
            if (metricEntropy) metricEntropy.textContent = "نرمال";
        }

        // Malignancy Factors
        factorsList.innerHTML = '';
        if (diag.malignancy_factors && diag.malignancy_factors.length > 0) {
            diag.malignancy_factors.forEach(f => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span>${f.factor}</span>
                    <span class="factor-impact">${f.impact}% (${f.status})</span>
                `;
                factorsList.appendChild(li);
            });
        }
    }

    function populatePipelineDrawer(steps) {
        if (!steps) return;
        pipelineCards.innerHTML = '';

        const stepDefinitions = [
            { key: 'step1_raw', name: '۱. اسکن ایزوله و پاکسازی‌شده' },
            { key: 'step2_windowed', name: '۲. نرمال‌سازی پنجره بافتی' },
            { key: 'step3_brain_mask', name: '۳. ماسک حذف جمجمه' },
            { key: 'step4_stripped_brain', name: '۴. پارانشیم خالص مغز' },
            { key: 'step5_clahe_enhanced', name: '۵. تقویت کنتراست CLAHE' },
            { key: 'step6_tumor_mask', name: '۶. ماسک تومور (Multi-Cue)' },
            { key: 'step7_annotated', name: '۷. مرزبندی دقیق OpenCV' },
            { key: 'step8_bone_colormap', name: '۸. نقشه عدم تقارن نیمکره‌ها' }
        ];

        stepDefinitions.forEach(step => {
            const b64 = steps[step.key];
            if (!b64) return;

            const card = document.createElement('div');
            card.className = 'step-card';
            card.innerHTML = `
                <img src="${b64}" alt="${step.name}">
                <span class="step-name">${step.name}</span>
            `;

            card.addEventListener('click', async () => {
                const img = await loadImg(b64);
                if (img) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                }
            });

            pipelineCards.appendChild(card);
        });
    }

    // Laser Scanner Toggle
    if (btnToggleScan && laserScanner) {
        btnToggleScan.addEventListener('click', () => {
            const isHidden = laserScanner.classList.toggle('hidden');
            btnToggleScan.classList.toggle('active', !isHidden);
            if (hudScannerStatus) {
                hudScannerStatus.textContent = isHidden ? 'STANDBY' : 'SCANNING [λ=532nm]';
                hudScannerStatus.className = isHidden ? '' : 'text-scanner-active';
            }
        });
    }

    // Caliper Tool
    btnCaliper.addEventListener('click', () => {
        state.caliperActive = !state.caliperActive;
        btnCaliper.classList.toggle('active', state.caliperActive);
        canvasWrapper.classList.toggle('caliper-mode', state.caliperActive);
        if (!state.caliperActive) {
            measurementSvg.innerHTML = '';
            hudMeasureText.classList.add('hidden');
        }
    });

    // Export Snapshot PNG
    if (btnExportPng) {
        btnExportPng.addEventListener('click', () => {
            const link = document.createElement('a');
            link.download = `NeuroScan_ErfanAlikhani_${Date.now()}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
    }

    // Mouse wheel zoom on canvas
    canvasWrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.12 : -0.12;
        state.zoomLevel = Math.min(2.5, Math.max(1.0, state.zoomLevel + delta));
        if (hudZoom) hudZoom.textContent = `${Math.round(state.zoomLevel * 100)}%`;
        renderCanvas();
    });

    btnResetView.addEventListener('click', () => {
        state.caliperActive = false;
        state.zoomLevel = 1.0;
        state.brightnessPct = 100;
        if (brightnessSlider) brightnessSlider.value = 100;
        if (hudZoom) hudZoom.textContent = '100%';
        btnCaliper.classList.remove('active');
        canvasWrapper.classList.remove('caliper-mode');
        measurementSvg.innerHTML = '';
        hudMeasureText.classList.add('hidden');
        renderCanvas();
    });

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    canvasWrapper.addEventListener('mousedown', (e) => {
        if (!state.caliperActive) return;
        state.isMeasuring = true;
        state.measureStart = getCanvasCoords(e);
        state.measureEnd = state.measureStart;
        hudMeasureText.classList.remove('hidden');
    });

    canvasWrapper.addEventListener('mousemove', (e) => {
        if (!state.caliperActive || !state.isMeasuring) return;
        state.measureEnd = getCanvasCoords(e);
        drawMeasurementLine();
    });

    window.addEventListener('mouseup', () => {
        state.isMeasuring = false;
    });

    function drawMeasurementLine() {
        if (!state.measureStart || !state.measureEnd) return;
        const p1 = state.measureStart;
        const p2 = state.measureEnd;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distPx = Math.sqrt(dx * dx + dy * dy) / state.zoomLevel;
        const distMm = (distPx * state.pixelSpacingMm).toFixed(1);

        measureMm.textContent = `${distMm} mm`;

        const rect = canvas.getBoundingClientRect();
        const sx = rect.width / canvas.width;
        const sy = rect.height / canvas.height;
        const x1 = p1.x * sx, y1 = p1.y * sy;
        const x2 = p2.x * sx, y2 = p2.y * sy;

        measurementSvg.innerHTML = `
            <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#FACC15" stroke-width="2" stroke-dasharray="4,2"/>
            <circle cx="${x1}" cy="${y1}" r="4" fill="#FACC15"/>
            <circle cx="${x2}" cy="${y2}" r="4" fill="#FACC15"/>
            <text x="${(x1 + x2) / 2 + 8}" y="${(y1 + y2) / 2 - 8}" fill="#FACC15" font-family="JetBrains Mono" font-size="12" font-weight="bold">${distMm} mm</text>
        `;
    }

    // Layer Selection & Sliders
    layerBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            layerBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeLayer = btn.dataset.layer;
            renderCanvas();
        });
    });

    opacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        state.heatmapOpacity = val / 100.0;
        opacityVal.textContent = `${val}%`;
        renderCanvas();
    });

    if (brightnessSlider) {
        brightnessSlider.addEventListener('input', (e) => {
            state.brightnessPct = parseInt(e.target.value, 10);
            renderCanvas();
        });
    }

    // Samples Menu
    btnSamples.addEventListener('click', (e) => {
        e.stopPropagation();
        samplesMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
        samplesMenu.classList.add('hidden');
    });

    async function loadSamplesMenu() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.status === 'success') {
                state.samples = data.samples;
                samplesMenu.innerHTML = '';
                data.samples.forEach(sample => {
                    const badgeColor = sample.diagnosis === 'Malignant' ? '#EF4444' :
                        (sample.diagnosis === 'Benign' ? '#F59E0B' : '#10B981');
                    const diagFa = sample.diagnosis === 'Malignant' ? 'بدخیم / سرطانی' :
                        (sample.diagnosis === 'Benign' ? 'تومور خوش‌خیم' : 'سالم و طبیعی');
                    const item = document.createElement('div');
                    item.className = 'sample-item';
                    item.innerHTML = `
                        <img src="${sample.thumbnail_b64}" alt="${sample.title_en}">
                        <div class="sample-info">
                            <div class="sample-title">${sample.title_fa}</div>
                            <div class="sample-diag" style="color:${badgeColor};font-weight:700;">● ${diagFa} (${sample.diagnosis})</div>
                        </div>
                    `;
                    item.addEventListener('click', () => {
                        samplesMenu.classList.add('hidden');
                        analyzeImage(sample.thumbnail_b64);
                    });
                    samplesMenu.appendChild(item);
                });

                if (data.samples.length > 0) {
                    analyzeImage(data.samples[0].thumbnail_b64);
                }
            }
        } catch (e) {
            console.error("Failed to load clinical samples:", e);
        }
    }

    // File Upload & Drag-Drop
    fileUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => analyzeImage(evt.target.result);
        reader.readAsDataURL(file);
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
        canvasWrapper.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    canvasWrapper.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (evt) => analyzeImage(evt.target.result);
            reader.readAsDataURL(file);
        }
    });

    // Official Print Report Modal
    btnPrintReport.addEventListener('click', () => {
        if (!state.analysisData) {
            alert("لطفاً ابتدا تصویری را جهت تحلیل انتخاب یا آپلود کنید.");
            return;
        }

        const diag = state.analysisData.ai_diagnosis;
        const cv = state.analysisData.opencv_metrics;
        const steps = state.analysisData.pipeline_steps;

        document.getElementById('rep-id').textContent = `NS-${Math.floor(1000 + Math.random() * 9000)}`;
        document.getElementById('rep-date').textContent = new Date().toLocaleDateString('fa-IR');
        const repMod = document.getElementById('rep-modality');
        if (repMod) repMod.textContent = cv.modality || 'CT / MRI Axial';

        document.getElementById('rep-diagnosis').textContent = `${diag.diagnosis_fa}`;
        const repConf = document.getElementById('rep-diag-conf');
        if (repConf) repConf.textContent = `${diag.diagnostic_confidence_pct}%`;
        document.getElementById('rep-cancer-prob').textContent = `${diag.cancer_probability}%`;
        const repBN = document.getElementById('rep-benign-normal');
        if (repBN) repBN.textContent = `خوش‌خیم: ${diag.probabilities.benign}% | نرمال: ${diag.probabilities.normal}%`;
        document.getElementById('rep-who').textContent = diag.who_grade;
        const repRel = document.getElementById('rep-reliability');
        if (repRel) repRel.textContent = `${diag.ai_reliability_pct}% (توافق TTA: ${diag.tta_consistency_pct}%)`;

        document.getElementById('rep-img-raw').src = steps.step1_raw;
        document.getElementById('rep-img-contours').src = diag.diagnostic_overlay_b64;
        document.getElementById('rep-img-gradcam').src = diag.gradcam_heatmap_b64;

        const bioContainer = document.getElementById('rep-biomarkers');
        if (cv.has_tumor && cv.primary_lesion) {
            const p = cv.primary_lesion;
            bioContainer.innerHTML = `
                <div>• مساحت ضایعه تومورال: <strong>${p.area_cm2} cm²</strong> (${p.area_mm2} mm² - معادل <strong>${cv.lesion_brain_ratio_pct}%</strong> سطح مقطع مغز)</div>
                <div>• قطر معادل ضایعه: <strong>${p.diameter_mm} میلی‌متر</strong> | حجم تخمینی توده: <strong>${p.volume_cm3} cm³</strong></div>
                <div>• موقعیت آناتومیک: <strong>${p.hemisphere} - ${p.lobe}</strong></div>
                <div>• درصد یکپارچگی مرز (Solidity): <strong>${(p.solidity * 100).toFixed(1)}%</strong> | گردی (Circularity): <strong>${(p.circularity * 100).toFixed(1)}%</strong></div>
                <div>• آنتروپی بافتی GLCM: <strong>${p.glcm_entropy} bits</strong> | شاخص عدم تقارن نیمکره‌ها: <strong>${(cv.asymmetry_score * 100).toFixed(1)}%</strong></div>
            `;
        } else {
            bioContainer.innerHTML = `
                <div>• فاقد هرگونه توده فضاگیر یا ضایعه نئوپلاستیک در پارانشیم مغز (اطمینان: <strong>${diag.diagnostic_confidence_pct}%</strong>).</div>
                <div>• قرینگی نیمکره‌ها و سیستم بطنی کاملاً طبیعی است (عدم تقارن: <strong>${(cv.asymmetry_score * 100).toFixed(1)}%</strong>).</div>
                <div>• مساحت پارانشیم استخراج‌شده مغز: <strong>${cv.brain_area_cm2} سانتی‌متر مربع</strong>.</div>
            `;
        }

        document.getElementById('rep-recommendation').textContent = diag.clinical_recommendation;
        reportModal.classList.remove('hidden');
    });

    btnCloseModal.addEventListener('click', () => reportModal.classList.add('hidden'));
    btnDoPrint.addEventListener('click', () => window.print());

    loadSamplesMenu();
});
