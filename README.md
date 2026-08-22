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
`--response-csv`, `--skip-umap-3d`. Run `python scripts/scrna_seq_for_lbcl.py --help` for the full list.

### 3D UMAP embedding CSV

Every run (unless `--skip-umap-3d` is passed) computes a 3D UMAP embedding of the full cohort
(GPU via RAPIDS cuML, falling back to CPU `umap-learn`, or skipped with a warning if neither is
installed) and saves it as `umap_3d.csv` (`UMAP_1/2/3`, `Cell_Type`, `Tumor_Response`, `Is_Outlier`,
`Patient`) in `--output-dir`. This only exports the data — no plot is rendered on the server — so you
can build the interactive 3D view yourself later, e.g.:

```python
import pandas as pd, plotly.express as px
df = pd.read_csv("umap_3d.csv")
fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3", color="Cell_Type")
fig.update_traces(marker=dict(size=2, opacity=0.7))
fig.show()
```

Tune with `--umap-n-neighbors` (default 15) and `--umap-min-dist` (default 0.1).

### Optional: real pretrained scGPT (transfer learning, opt-in)

The four benchmark architectures are compact "-Inspired" networks trained from scratch. Passing
`--use-pretrained-scgpt` adds a genuine 5th model: the real [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
foundation-model checkpoint, fine-tuned on this cohort. This is a best-effort integration written
against the scGPT fine-tuning tutorial pattern; it has **not** been runtime-tested against a real
checkpoint (no GPU/`scgpt`/checkpoint were available while writing it). If it errors, please share
the traceback so the call can be pinned to your installed `scgpt` version.

**Setup — run these yourself before using `--use-pretrained-scgpt` (not run by the script):**

```bash
# 1. Clone the official scGPT repo
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT

# 2. Install scGPT and its dependencies (Python 3.9/3.10 + a CUDA-enabled torch build recommended)
pip install scgpt
# -- or, to install from the freshly cloned source instead of PyPI:
# pip install -e .

# 3. Download a pretrained checkpoint from the "Pretrained scGPT Model Zoo" table in the scGPT
#    repo's README (e.g. the "whole-human" checkpoint) and unzip it locally. The folder must
#    contain exactly: args.json, vocab.json, best_model.pt
cd ..
```

**Run the pipeline pointing at both locations:**

```bash
python scripts/scrna_seq_for_lbcl.py \
  --use-pretrained-scgpt \
  --scgpt-repo-dir ./scGPT \
  --scgpt-checkpoint-dir /path/to/unzipped/checkpoint \
  --response-csv patient_response.csv --skip-download
```

Other flags: `--scgpt-freeze-backbone` (linear-probe: freeze the pretrained transformer, train only
the new classification heads; faster and lower memory) and `--scgpt-n-bins` (expression-value
tokenizer bins, default 51, matching the original pretraining).

**Iterating quickly:** add `--scgpt-only` to skip the four from-scratch architectures and the
Optuna search entirely, and benchmark *only* the pretrained model (useful while debugging the
integration, since you don't want to wait through the other four every time). Requires
`--use-pretrained-scgpt`; fails fast with a clear message if the checkpoint doesn't load, rather
than silently falling back to the other four.

```bash
python scripts/scrna_seq_for_lbcl.py \
  --use-pretrained-scgpt --scgpt-only \
  --scgpt-repo-dir ./scGPT \
  --scgpt-checkpoint-dir /path/to/unzipped/checkpoint \
  --response-csv patient_response.csv --skip-download
```

**Known limitation:** only the pretrained model's own gene vocabulary can be tokenized, so HVGs
absent from it are dropped for this model only (a match-rate is printed at load time), and
expression values are rank-binned per cell (robust to our already-z-scored input) rather than
reproducing scGPT's official magnitude-based binning of raw counts — a pragmatic simplification,
not an exact reproduction of the original preprocessing.

### Labels (important)

Cell labels are grounded in the real cohort, not random:
- **CAR-T** = cells from the FACS `CAR-pos` sorted samples (`sorting: CAR-pos` in GEO, `...-CART` titles).
- **B vs T** = canonical lineage markers (CD3D/E/G; MS4A1/CD79A/B/CD19).
- **Clinical response** = per-patient responder/non-responder. GEO does **not** encode response, so you
  must supply it via `--response-csv` (columns `patient,response`), filled from Haradhvala et al. 2022
  Supplementary Table 1 (responder = no relapse by 6 months). Run once without the flag to get a blank
  `patient_response_template.csv` listing every patient. `--random-response` exists for pipeline smoke
  tests only and produces scientifically invalid resistance results.

### Evaluation protocol (avoid leakage)

Because clinical response is a **patient-level** label, all splitting is by patient (`--split-by patient`,
default) so no patient appears in both train and test. A naive per-cell split (`--split-by cell`) leaks
patient identity and inflates resistance F1 roughly two-fold. For a robust estimate over the small
(32-patient) cohort, use **GroupKFold cross-validation**, which holds out whole patients per fold and
reports mean±SD across folds:

```bash
python scripts/scrna_seq_for_lbcl.py --response-csv patient_response.csv --cv-folds 5 --skip-download
```

## Data

The GSE197268 raw data (~GBs) is **not** included; the script downloads it from NCBI GEO at runtime.

## License

Code is released under the [MIT License](LICENSE).

## Citation

Primary dataset: Haradhvala, N. J., Leick, M. B., Maurer, K., et al. (2022). Distinct cellular
dynamics associated with response to CAR-T therapy for refractory B cell lymphoma.
*Nature Medicine*, 28(9), 1848–1859. https://doi.org/10.1038/s41591-022-01959-0
