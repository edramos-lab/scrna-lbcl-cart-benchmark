# Multi-Task Benchmarking of Single-Cell Foundation Models for CAR-T Resistance Prediction in Large B-Cell Lymphoma

A Pareto-optimal, energy-aware framework that benchmarks a compact multi-task baseline against
three single-cell foundation-model architectures (scBERT, scGPT, Geneformer) on joint
cell-lineage classification and CAR-T resistance prediction, using the real-world **GSE197268**
CAR-T clinical atlas (Haradhvala et al., *Nature Medicine* 2022).

## Contents

| Path | Description |
|------|-------------|
| `images/` | Figures (Green AI Pareto, t-SNE, XAI, power traces) |
| `scripts/scrna_seq_for_lbcl.py` | End-to-end CLI pipeline (data → benchmark → Optuna HPO → XAI) |

> The manuscript (LaTeX source, bibliography, and compiled PDF) is maintained separately and is
> not tracked in this repository.

## Pipeline script

The script downloads and ingests the GSE197268 10x Genomics matrices with Scanpy, benchmarks the
four architectures (Macro-F1, latency, energy), runs an Optuna hyperparameter search, samples real
GPU power during training (NVML), and produces GPU-accelerated 2D t-SNE, Isolation-Forest anomaly,
and Integrated-Gradients explainability figures.

### Run in Google Colab (GPU runtime)

```bash
pip install -q optuna nvidia-ml-py scanpy anndata captum

python scripts/scrna_seq_for_lbcl.py \
  --gpu-power-watts 400 \
  --power-monitor \
  --tune-model "scGPT (Generative)" \
  --n-trials 30 --search-epochs 20 \
  --response-csv patient_response.csv \
  --output-dir ./outputs
```

Key flags: `--gpu-power-watts` (T4=70, V100=300, A100=400), `--tune-model`, `--n-trials`,
`--skip-download` (reuse data on re-runs), `--power-monitor`, `--skip-gpu-tsne`,
`--response-csv`. Run `python scripts/scrna_seq_for_lbcl.py --help` for the full list.

### Labels (important)

Cell labels are grounded in the real cohort, not random:
- **CAR-T** = cells from the FACS `CAR-pos` sorted samples (`sorting: CAR-pos` in GEO, `...-CART` titles).
- **B vs T** = canonical lineage markers (CD3D/E/G; MS4A1/CD79A/B/CD19).
- **Clinical response** = per-patient responder/non-responder. GEO does **not** encode response, so you
  must supply it via `--response-csv` (columns `patient,response`), filled from Haradhvala et al. 2022
  Supplementary Table 1 (responder = no relapse by 6 months). Run once without the flag to get a blank
  `patient_response_template.csv` listing every patient. `--random-response` exists for pipeline smoke
  tests only and produces scientifically invalid resistance results.

## Data

The GSE197268 raw data (~GBs) is **not** included; the script downloads it from NCBI GEO at runtime.

## License

Code is released under the [MIT License](LICENSE).

## Citation

Primary dataset: Haradhvala, N. J., Leick, M. B., Maurer, K., et al. (2022). Distinct cellular
dynamics associated with response to CAR-T therapy for refractory B cell lymphoma.
*Nature Medicine*, 28(9), 1848–1859. https://doi.org/10.1038/s41591-022-01959-0
