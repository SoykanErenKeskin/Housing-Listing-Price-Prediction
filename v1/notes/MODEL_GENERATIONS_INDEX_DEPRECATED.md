# Model Generations Index

## V1 — Thesis Legacy

Path:
`model_generations/v1_thesis_legacy/`

Scope:
Tez dönemi ve klasik Kocaeli modeli.

Includes:
v1–v16 (incl. `v6.1 tez sonu`, `v9.1`)

Best known checkpoint:
V15/V16 civarı, **manual review required**.

Known issue:
Başiskele R2 düşük ve variance compression problemi devam ediyor.

---

## V2 — Location + Başiskele Sandbox

Path:
`model_generations/v2_location_basiskele/`

Scope:
V17 lokasyon feature dönemi + V18 Başiskele-only comparable denemeleri.

Includes:
`v17`, `v18_basiskele`

Best known checkpoint:
V18 Başiskele-only geo control:

- `comparable_mode` none
- R2 approx 0.4731
- MAPE approx 0.1093
- variance_ratio approx 0.4264

Also archived:
V17 Kocaeli location full run under `best_kocaeli_location_checkpoint/`.

Rejected:

- nearest comparable
- similar comparable
- weighted comparable
- full comparable

Known issue:
Mean-pulling / expensive underprediction devam ediyor.

---

## V3 — Next Experiments

Path:
`model_generations/v3_next_experiments/`

Scope:
V19+ calibration / anti-shrink / target-profile / ensemble-profile deneyleri.

Status:
Scaffold only.

---

## Safety notes

- Root `v1`…`v18_basiskele` klasörleri **silinmedi**.
- `model_generations/` çoğunlukla **kopya + index**; büyük `*.joblib` çoğunlukla path referansı.
- Eğitim scriptleri / feature logic değiştirilmedi.
