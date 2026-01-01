# Immunotherapy Response Predictor

A unified pipeline for extracting immune features from tumor gene expression data and predicting immunotherapy response.

## Features

- **Feature Extraction Pipeline**: Extracts immune features using multiple established methods:
  - **ImmuCellAI**: Immune cell deconvolution (53 cell types)
  - **TIDE**: Tumor Immune Dysfunction and Exclusion scoring
  - **ImmuneSubtypes**: Cancer immune subtype classification (6 subtypes)
  - **IMPRES**: Immunotherapy response predictor based on gene pairs

- **ML Predictors**: Train models for immunotherapy response prediction:
  - XGBoost
  - Random Forest
  - SVM
  - Logistic Regression

## Installation

```bash
# Clone the repository
git clone https://github.com/Anxxsh7/Immunotherapy-Predictor.git
cd Immunotherapy-Predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### R Dependencies

For IMPRES and ImmuneSubtypes, you need R with the following packages:
```r
install.packages("data.table")
# For ImmuneSubtypeClassifier, see: https://github.com/Gibbsdavidl/ImmuneSubtypeClassifier
```

## Usage

### Feature Extraction

```bash
python feature_pipeline.py --input expression_tpm.txt --output features.csv --cancer-type Melanoma
```

**Input Format**: TPM expression matrix (genes as rows, samples as columns, tab-separated)

### Training Predictors

```bash
python Predict/train_xgboost_predictor.py
```

## Directory Structure

```
├── feature_pipeline.py      # Main feature extraction pipeline
├── Predict/                  # ML training scripts
├── TIDE/                     # TIDE analysis and plotting scripts
├── Datasets/                 # Expression data and clinical info
│   ├── GSE91061/            # Riaz et al. anti-PD-1 dataset
│   └── GSE160638/           # PD-1 blockade melanoma dataset
├── IMMUCELL.py              # ImmuCellAI wrapper
├── IMPRES.R                 # IMPRES calculation
└── IMMUNESUBTYPES.R         # Immune subtype classification
```

## References

- **TIDE**: Jiang et al., Nature Medicine, 2018
- **ImmuCellAI**: Miao et al., Cancer Research, 2020
- **IMPRES**: Auslander et al., Nature Medicine, 2018
- **ImmuneSubtypes**: Thorsson et al., Immunity, 2018

## License

MIT License - see [LICENSE](LICENSE) for details.
