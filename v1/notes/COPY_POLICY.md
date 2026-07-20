# Copy policy (V1 archive)

- Root `v1`…`v16` **silinmedi**, sadece kopyalandı.
- `source_versions/` kopyasında **hariç**: `outputs/`, `__pycache__/`, `.env`, `*.joblib`.
- `reports/` altına v14/v15/v16 full report’lar ve root `outputs/v13_*`, `outputs/v16_diagnostics` kopyalandı.
- Full `data/` CSV dump’ları ve multi-GB artifact bundle’lar **kopyalanmadı** (boyut).
- Orijinal artifact yolları: `artifacts/ARTIFACT_PATH_REFERENCE.md`.

## Not copied due to size

- `v14/outputs/**/artifacts/*.joblib`
- `v15/outputs/**/artifacts/*.joblib`
- `v16/outputs/**/artifacts/*.joblib`
- Full train `data/` folders under those runs (hundreds of MB)

## Unclassified

Emin olunamayan / dönem dışı kalan root çıktıları varsa `unclassified/` veya bu notta listelenir.
Şu an root `outputs/` içindeki v17_test V2’ye alındı; v13/v16 thesis-era burada.
