# Data Licenses — Roof Corrosion MLOps

> **Rule**: No NC-licensed weights or data may be included in production model artifacts.
> Research-only weights may be used for pretraining but must be fine-tuned on
> commercially-safe data before promotion to `staging` or `production` in MLflow.

| Dataset | GSD | Labels | License | Commercial? | Usage in Project | Notes |
|---|---|---|---|---|---|---|
| **Open AI Caribbean Challenge** | 5–10 cm (drone) | Roof type (healthy/irregular metal) | CC-BY-4.0 | ✅ Yes | Phase 1a: corrosion proxy baseline | `irregular_metal` ≈ corroded metal roof |
| **AIRS (Aerial Imagery Roof Segmentation)** | 7.5 cm | Roof masks | Research-only¹ | ⚠️ Research | Phase 1a: Stage-1 roof pretrain | Check Christchurch CC terms; may need license for prod |
| **SpaceNet 1–8** | 30–50 cm (Maxar) | Building polygons | CC-BY-SA-4.0 | ✅ Yes | Phase 1a: Maxar-domain roof pretrain | Best generic pretraining for Maxar GSD |
| **xBD (xView2)** | 30–80 cm (Maxar) | Building damage (4 classes) | CC-BY-NC-4.0 | ❌ No | Phase 1a: research-only backbone pretrain | **Must NOT ship in prod weights.** Fine-tune off, then retrain on safe data. |
| **Inria Aerial Labeling** | 30 cm | Building masks | Research-only | ⚠️ Research | Phase 1a: alt roof pretrain | Clean, well-benchmarked |
| **Open Cities AI** | 5–20 cm (drone) | Building polygons | Open | ✅ Yes | Phase 1a: alt roof pretrain | African cities, drone-quality |
| **Microsoft Global Building Footprints** | — | Polygons only (no imagery) | ODbL-1.0 | ✅ Yes | Phase 1b+: free ground truth for any tile | No imagery needed |
| **Google Open Buildings** | — | Polygons, Africa/S-Asia/LatAm | CC-BY-4.0 | ✅ Yes | Phase 1b+: free ground truth | Different coverage than MSFT |
| **LandCover.ai** | 25–50 cm (Poland) | Land cover incl. buildings | CC-BY-NC-SA | ❌ No | Not used | NC-SA incompatible with commercial |
| **NEU Surface Defect** | Ground-level | Steel defect classes | Research | ⚠️ Research | Texture pretraining only | Severe domain gap to overhead |
| **Severstal Steel Defect (Kaggle)** | Ground-level | Steel defect masks | Kaggle Rules² | ⚠️ Check | Texture pretraining only | Severe domain gap to overhead |
| **Own "normal image" corrosion dataset** | Ground-level | Corrosion labels | Proprietary | ✅ Yes | Ground-level texture pretrain | Does NOT transfer to overhead; separate product path |

¹ AIRS license is ambiguous — contact Canterbury University / LINZ for commercial clarification.
² Kaggle competition datasets typically restrict commercial use; check specific competition rules.

---

## Enforcement

1. Every MLflow run must tag `data_sources` with the list of datasets used.
2. MLflow promotion gate (`dev → staging → production`) checks tags:
   - If any NC / research-only source is tagged → **block promotion**.
3. CI step: `python ml/data/check_licenses.py` validates no NC data in training manifest.
4. Audit trail: every shipped model version has a `DATA_PROVENANCE.md` artifact in MLflow.
