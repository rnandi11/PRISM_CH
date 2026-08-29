<h2>PRISM - A machine learning model using routine blood count data to prioritize sequencing for high-risk clonal hematopoiesis</h2>
This study used routine blood counts and demographic data from UK Biobank to train a Balanced Random Forest model to predict high-risk Clonal Hematopoiesis. It takes participant's age and routine blood counts as input features and classifies them in one of the three groups:

**`Positive`** — directs individuals to molecular profiling with sequencing to screen for the presence of CH.
**`Negative`** — indicates that the individual is highly unlikely to harbor high-risk CH and is unlikely to benefit from screening for CH.
**`Equivocal`** — an indeterminant class for whom single-time point CBCs are insufficient to determine the need for CH screening. Follow-up assessments are recommended.


<h3> Python Scipts </h3>

1. `baseline_model.py` - This python script can be independently run to generate the baseline model from the preprocessed dataset.
2. `final_model.py` - This python script can be run to generate the final model from the baseline model (after threshold selection, and refitting on final feature set) on which performance is tested.
3. `PRISM_predict_evaluate.py` - This script generates PRISM predictions on an input dataset, creating a new output file with predicted probabilities of PRISM output classes and predicted class labels. If ground truth labels exist in the input data, then model performance is evaluated against those labels. Figures and files of model performance metrics and files with CHRS benchmarking is generated.
4. `prism_imputation.py` - This script implements KNN imputation for handling missing data using five nearest neighbors (k=5). KNN imputation may not be appropriate for features with a high proportion of missing values, as there may be insufficient information to produce reliable estimates. Place this file in the same directory as 'PRISM_predict_evaluate_w_imputation.py', which imports and calls the 'impute_missing_features' function.
5. `PRISM_predict_evaluate_w_imputation.py` - This script provides the same functionality as `PRISM_predict_evaluate.py` with additional support for handling missing data through imputation.

## Pipeline Usage

Run the following scripts in order. Each stage produces outputs required by the next.

### 1. (Optional) Impute missing blood count data
```bash
python3 impute_data.py
```
- Imputes missing values in the raw blood count data (via KNN (K=5) imputation)
- Produces the cleaned/imputed dataset used by all downstream steps

**Before running:** update the input/output file paths at the top of the script.

**Output:**
- `df_imputed.csv` — imputed dataset used as input to Step 2


### 2. Prepare input data
Place `xxx.csv` (either imputed or original dataset) in the project root (or update the path inside the script).
The dataset must contain the following columns (or `PDW`, from which `giant_plt` will be derived):

### 3. Train the baseline model
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
- `models/baseline_custom_threshold.pkl` — model, features, thresholds

### 4. Retrain final model with OOF thresholds
```bash
python3 final_model.py
```
- Loads baseline model hyperparameters from `models/baseline_model.pkl`
- Generates out-of-fold probabilities via 5-fold CV on the full training set
- Derives per-class optimal thresholds (Youden's J) from OOF predictions
- Restricts features to the deployment feature set
- Refits the final model and evaluates on the held-out test set

**Before running:** update `TRAIN_CSV` and `TEST_CSV` paths at the top of the script.

**Outputs:**
- `models/final_model.pkl` — final model, OOF thresholds, feature list, class labels

### 5. Generates PRISM predictions and evaluates model performance
Choose the appropriate workflow based on whether your dataset contains missing values.  

**Without missing-data imputation:**
```bash
python3 PRISM_predict_evaluate.py
```
**With missing-data imputation:**
```bash
python3 PRISM_predict_evaluate_w_imputation.py
```
- Both scripts generate PRISM predictions and evaluate model performance on the test data.
- Applies the same feature construction (`giant_plt` from `PDW` if needed) and threshold-based prediction logic
- PRISM_predict_evaluate_w_imputation.py handles missing feature values using the impute_missing_features function from prism_imputation.py.

**Before running:** update the model path (`path/to/model.pkl`) and validation data path (`path/to/data.csv`) at the top of the script.

**Outputs:**
- `output_prism_predictions.csv`
- `roc_auc.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `chrs_statistics.txt`

### Notes
- Random seed fixed at `345` throughout for reproducibility.
- Scripts expect a `models/` directory for saved `.pkl` files — create it if it doesn't exist, or let `save_model()` create it automatically.
