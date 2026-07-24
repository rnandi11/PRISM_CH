<h2>PRISM - A machine learning model using routine blood count data to prioritize sequencing for high-risk clonal hematopoiesis</h2>
This study used routine blood counts and demographic data from UK Biobank to train a Balanced Random Forest model to predict high-risk Clonal Hematopoiesis. 


<h3> Python Scipts </h3>

1. `baseline_model.py` - This python script can be independently run to generate the baseline model from the preprocessed dataset.
2. `final_model.py` - This python script can be run to generate the final model from the baseline model (after threshold selection, and refitting on final feature set) on which performance is tested.
3. `impute_data.py` - Script for implementing KNN imputation for missing data.
4. `performance_evaluation.py` - This script generates figures and files of model performance metrics and CHRS benchmarking, and can be used on external validation sets.

## Pipeline Usage

Run the following scripts in order. Each stage produces outputs required by the next.

### 1. Prepare input data
Place `xxx.csv` in the project root (or update the path inside the script).
The dataset must contain the following columns (or `PDW`, from which `giant_plt` will be derived):

### 2. Train the baseline model
```bash
python3 baseline_model.py
```
- Trains a `BalancedRandomForestClassifier` with `RandomizedSearchCV`
- Runs Boruta feature selection and retrains on selected features
- Evaluates the model (standard + custom threshold-based predictions)
- Runs SHAP analysis per class

**Outputs:**
- `training_df.csv`, `test_df.csv` — train/test splits
- `boruta_results.txt` — selected features and rankings
- `model_results.txt` — accuracy, classification report, thresholds
- `baseline_confusion_matrix_standard.png`, `baseline_confusion_matrix_custom_threshold.png`
- `baseline_roc_auc.png`
- `shap_summary_class0.png`, `shap_summary_class1.png`, `shap_summary_class2.png`
- `models/baseline_model.pkl` — model, features, thresholds

### 3. Retrain final model with OOF thresholds
```bash
python3 train_pipeline.py
```
- Loads baseline model hyperparameters from `models/baseline_model.pkl`
- Generates out-of-fold probabilities via 5-fold CV on the full training set
- Derives per-class optimal thresholds (Youden's J) from OOF predictions
- Restricts features to the deployment feature set
- Refits the final model and evaluates on the held-out test set

**Before running:** update `TRAIN_CSV` and `TEST_CSV` paths at the top of the script.

**Outputs:**
- `models/final_model.pkl` — final model, OOF thresholds, feature list, class labels

### 4. Validate on an external/holdout dataset
```bash
python3 validate_model.py
```
- Loads `models/final_model.pkl`
- Applies the same feature construction (`giant_plt` from `PDW` if needed) and threshold-based prediction logic
- Evaluates performance and computes CHRS/MN-based risk stratification statistics

**Before running:** update the model path (`path/to/model.pkl`) and validation data path (`path/to/data.csv`) at the top of the script.

**Outputs:**
- `roc_auc.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `chrs_statistics.txt`

### Notes
- Random seed fixed at `345` throughout for reproducibility.
- Scripts expect a `models/` directory for saved `.pkl` files — create it if it doesn't exist, or let `save_model()` create it automatically.
