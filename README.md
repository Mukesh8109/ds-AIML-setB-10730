# Demand Analysis and Preprocessing Pipeline

This repository contains a Jupyter Notebook focused on synthetic dataset generation, exploratory data analysis, data quality handling, and modular processing pipelines for a binary classification task.

## Project Structure

* **Data Generation:** Simulates structural metrics including `temperature`, `occupancy`, `runtime`, and `load` split across groups (`G1`, `G2`) to determine a binary target indicator (`high_demand`).
* **Exploratory Data Analysis (EDA):** Statistical tests (such as Welch's t-test) and dimensionality reduction (Principal Component Analysis via covariance matrices).
* **Data Preprocessing & Splitting:** Implements clean data partitioning (`fit`, `validation`, and `test` stratifications) and addresses missing value treatments.
* **Evaluation Framework:** Validates classification quality tracking metrics using structured confusion matrices (`TP`, `FP`, `TN`, `FN`).

---

## Getting Started

### Prerequisites

Ensure you have the core scientific stack installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Running the Notebook

1. Clone or download this project.
2. Launch your preferred interface (Jupyter Notebook, JupyterLab, or Google Colab).
3. Open and run the `.ipynb` notebook file sequentially.

---

## Detailed Pipeline Breakdown

### 1. Data Generation & Initial Inspection
* Generates a structural base DataFrame (`300` records) with dynamic variance adjustments.
* Intentionally inserts localized missing entries (`NaN`) into critical telemetry tracking paths (`temperature`, `occupancy`) to test downstream robustness.
* Exports raw snapshots into the file system path: `data/raw/set_e.csv`.

### 2. Analytical Sub-Tasks (Task 1)
* **Statistical Profiling:** Evaluates distributions on the `runtime` parameters through visualization (Histograms) and measures deviations.
* **Hypothesis Testing:** Executes a Welch's t-test between group behaviors (`G1` vs. `G2`) on runtime output parameters to flag structural variance disparities.
* **Dimensionality Reduction:** Processes un-scaled variances using an Eigh-decomposition (`np.linalg.eigh`) over a centered covariance matrix to extract principal direction components.

### 3. Data Cleansing & Splitting Strategy (Task 2)
* Isolates and removes duplicate entries to safeguard model cross-validation.
* Segregates data into exact structural matrices:
  * **Fit Set:** 80% of training allocation -> further segmented into dedicated fit tracking parameters (192 rows).
  * **Validation Set:** Structural optimization validation threshold (48 rows).
  * **Test Set:** Final un-biased performance evaluation step (60 rows).
* Employs scikit-learn transformers (`SimpleImputer`) utilizing a **median strategy** calculation computed purely from training properties to fill missing cell slots uniformly across validation collections.

### 4. Metric Foundations (Task 3)
* Houses verification logic parsing structural categorical labels against predictive output sequences.
* Unpacks matrix dimensions cleanly via `.ravel()` into `tn`, `fp`, `fn`, `tp` formats to secure foundational arithmetic validation blocks before deploying scale classifiers.
