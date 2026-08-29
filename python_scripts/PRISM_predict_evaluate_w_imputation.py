import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from prism_imputation import impute_missing_features


# ======================== USER SETTINGS ========================
MODEL_PATH = "model.pkl"
DATA_PATH = "path/to/input_file.csv"
OUTPUT_PATH = "output_prism_predictions.csv"

CLASS_LABELS = {
    0: "negative",
    1: "equivocal",
    2: "positive",
}
# ========================================================================


with open(MODEL_PATH, "rb") as f:
    loaded_package = pickle.load(f)

loaded_model = loaded_package["model"]
loaded_features = loaded_package["features"]
loaded_thresholds = loaded_package["optimal_thresholds"]
loaded_classes = np.asarray(loaded_package["classes"])

print("Features used in model training:", loaded_features)


def plot_roc_auc(model, X, y, classes, save_path="roc_auc.png"):
    """Plot and save one-vs-rest ROC curves and return probabilities/AUCs."""
    y_bin = label_binarize(y, classes=classes)
    y_pred_proba = model.predict_proba(X)

    plt.figure(figsize=(5, 4))
    colors = ["blue", "red", "green", "purple", "orange"]
    auc_scores = {}

    for i, cls in enumerate(classes):
        # ROC is undefined when this validation set has no positive or no
        # negative examples for a class.
        if np.unique(y_bin[:, i]).size < 2:
            print(f"Skipping ROC for class {cls}: both outcomes are not present.")
            continue

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        class_name = CLASS_LABELS.get(cls, str(cls))
        plt.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_name} (AUC={roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-Class ROC Curve", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {"auc_scores": auc_scores, "proba": y_pred_proba}


def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """Plot and save a confusion matrix using readable PRISM class names."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    display_labels = [CLASS_LABELS.get(cls, str(cls)) for cls in classes]
    cmap = sns.color_palette("Blues", as_cmap=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=cmap, ax=ax, values_format="d", text_kw={"fontsize": 10})
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_chrs_statistics(test_df, y_pred, output_file="chrs_statistics.txt"):
    """Compute CHRS/MN statistics for predicted numeric class labels."""
    required = {"CHRS", "MN"}
    missing = sorted(required.difference(test_df.columns))
    if missing:
        raise ValueError(
            "save_chrs_statistics requires these additional columns: "
            f"{missing}"
        )

    hirisk_true = (test_df["CHRS"] >= 12.5).sum()
    intermediate_true = (
        (test_df["CHRS"] >= 10) & (test_df["CHRS"] < 12.5)
    ).sum()
    lowrisk_true = ((test_df["CHRS"] > 0) & (test_df["CHRS"] < 10)).sum()
    norisk_true = (test_df["CHRS"] == 0).sum()

    high_true_with_mn = (
        (test_df["CHRS"] >= 12.5) & (test_df["MN"] == 1)
    ).sum()
    intermediate_true_with_mn = (
        (test_df["CHRS"] >= 10)
        & (test_df["CHRS"] < 12.5)
        & (test_df["MN"] == 1)
    ).sum()

    predicted = {
        cls: test_df.iloc[np.where(np.asarray(y_pred) == cls)[0]]
        for cls in (0, 1, 2)
    }

    def get_risk_counts(frame):
        return {
            "high": (frame["CHRS"] >= 12.5).sum(),
            "intermediate": (
                (frame["CHRS"] >= 10) & (frame["CHRS"] < 12.5)
            ).sum(),
            "low": ((frame["CHRS"] > 0) & (frame["CHRS"] < 10)).sum(),
            "none": (frame["CHRS"] == 0).sum(),
        }

    risks = {cls: get_risk_counts(frame) for cls, frame in predicted.items()}

    ch_mn = ((test_df["CHRS"] > 0) & (test_df["MN"] == 1)).sum()
    ch_no_mn = ((test_df["CHRS"] > 0) & (test_df["MN"] == 0)).sum()
    no_ch_mn = ((test_df["CHRS"] == 0) & (test_df["MN"] == 1)).sum()
    no_ch_no_mn = ((test_df["CHRS"] == 0) & (test_df["MN"] == 0)).sum()

    mn_pred2 = ((predicted[2]["CHRS"] > 0) & (predicted[2]["MN"] == 1)).sum()
    mn_pred1 = ((predicted[1]["CHRS"] > 0) & (predicted[1]["MN"] == 1)).sum()

    high_pred2_with_mn = (
        (predicted[2]["CHRS"] >= 12.5) & (predicted[2]["MN"] == 1)
    ).sum()
    high_pred1_with_mn = (
        (predicted[1]["CHRS"] >= 12.5) & (predicted[1]["MN"] == 1)
    ).sum()
    intermediate_pred2_with_mn = (
        (predicted[2]["CHRS"] >= 10)
        & (predicted[2]["CHRS"] < 12.5)
        & (predicted[2]["MN"] == 1)
    ).sum()
    intermediate_pred1_with_mn = (
        (predicted[1]["CHRS"] >= 10)
        & (predicted[1]["CHRS"] < 12.5)
        & (predicted[1]["MN"] == 1)
    ).sum()

    with open(output_file, "w") as f:
        f.write("=== CHRS AND MN STATISTICS ===\n\n")
        f.write("Actual Risk Distribution:\n")
        f.write(f"  High-risk: {hirisk_true}\n")
        f.write(f"  Intermediate-risk: {intermediate_true}\n")
        f.write(f"  Low-risk: {lowrisk_true}\n")
        f.write(f"  No-risk: {norisk_true}\n\n")

        f.write("Predicted Class Counts:\n")
        f.write(f"  Positive (class 2): {len(predicted[2])}\n")
        f.write(f"  Equivocal (class 1): {len(predicted[1])}\n")
        f.write(f"  Negative (class 0): {len(predicted[0])}\n\n")

        f.write("Risk Distribution Among Predictions:\n")
        for cls in (2, 1, 0):
            counts = risks[cls]
            f.write(f"  {CLASS_LABELS[cls].title()} (class {cls}):\n")
            for risk_name, count in counts.items():
                f.write(f"    {risk_name.title()}-risk: {count}\n")
            f.write("\n")

        f.write("=== MN-SPECIFIC COUNTS ===\n")
        f.write(f"Actual high-risk with MN: {high_true_with_mn}\n")
        f.write(f"  In predicted class 2: {high_pred2_with_mn}\n")
        f.write(f"  In predicted class 1: {high_pred1_with_mn}\n\n")
        f.write(
            f"Actual intermediate-risk with MN: {intermediate_true_with_mn}\n"
        )
        f.write(f"  In predicted class 2: {intermediate_pred2_with_mn}\n")
        f.write(f"  In predicted class 1: {intermediate_pred1_with_mn}\n\n")
        f.write("CHRS vs MN Distribution in Test Data:\n")
        f.write(f"  CH+ with MN: {ch_mn}\n")
        f.write(f"  CH+ without MN: {ch_no_mn}\n")
        f.write(f"  CH- with MN: {no_ch_mn}\n")
        f.write(f"  CH- without MN: {no_ch_no_mn}\n\n")
        f.write("Predicted CH+ with MN Individuals in Positive Classes:\n")
        f.write(f"  In class 2: {mn_pred2}\n")
        f.write(f"  In class 1: {mn_pred1}\n")

    print(f"Results saved to '{output_file}'")


def predict_with_thresholds(model, X, classes, thresholds):
    """Return probabilities and threshold-adjusted numeric predictions."""
    probabilities = model.predict_proba(X)
    threshold_array = np.asarray([thresholds[cls] for cls in classes])
    predictions = []

    for probability_vector in probabilities:
        passing = np.where(probability_vector >= threshold_array)[0]
        if passing.size:
            margins = probability_vector[passing] - threshold_array[passing]
            chosen_index = passing[np.argmax(margins)]
        else:
            chosen_index = np.argmax(probability_vector)
        predictions.append(classes[chosen_index])

    return probabilities, np.asarray(predictions)


def main():
    df = pd.read_csv(DATA_PATH)
    # Preserve the file exactly as supplied for the prediction output. Feature
    # engineering below is performed only on the model-input working copy.
    original_df = df.copy()

    if "giant_plt" not in df.columns:
        if "PDW" not in df.columns:
            raise ValueError(
                "Neither 'giant_plt' nor 'PDW' was found; giant_plt cannot be built."
            )
        df["giant_plt"] = np.where(df["PDW"] > 16.8, 1, 0)

    missing_features = [name for name in loaded_features if name not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required input features: {missing_features}")

    X = df[loaded_features].copy()

    if X.isna().any().any():
        print("Missing model-feature values detected. Running KNN imputation.")
        X = impute_missing_features(X, loaded_features)

    if X.isna().any().any():
        remaining = X.isna().sum()
        remaining = remaining[remaining > 0]
        raise ValueError(
            "Missing values remain after imputation:\n"
            f"{remaining.to_string()}"
        )


    # =================== ALWAYS GENERATE PREDICTIONS ===================
    y_proba, y_pred_numeric = predict_with_thresholds(
        loaded_model,
        X,
        loaded_classes,
        loaded_thresholds,
    )

    # Keep the original data and add the readable prediction column.
    output_df = original_df.copy()
    output_df["PRISM prediction"] = pd.Series(
        y_pred_numeric, index=output_df.index
    ).map(CLASS_LABELS)
    if output_df["PRISM prediction"].isna().any():
        unknown = sorted(set(y_pred_numeric).difference(CLASS_LABELS))
        raise ValueError(f"No text-label mapping is defined for model classes: {unknown}")


    # ================= ADD CLASS PROBABILITIES=================
    # predict_proba columns follow loaded_classes, so locate each class
    # explicitly instead of assuming the columns are always ordered 0, 1, 2.
    class_to_probability_index = {
        cls: index for index, cls in enumerate(loaded_classes)
    }
    for cls, label in CLASS_LABELS.items():
        if cls not in class_to_probability_index:
            raise ValueError(
                f"Model classes do not contain class {cls} ({label})."
            )
        output_df[f"prob_{label}"] = y_proba[
            :, class_to_probability_index[cls]
        ]
    # =====================================================================



    output_directory = os.path.dirname(OUTPUT_PATH)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to '{OUTPUT_PATH}'")
    # ============================================================================

    # ================= EVALUATE ONLY WHEN CH EXISTS =================
    if "CH" not in df.columns:
        print("No 'CH' column found; prediction file created without evaluation.")
        return

    y_true = df["CH"]
    if y_true.isna().any():
        raise ValueError("The 'CH' column contains missing ground-truth labels.")
    if not set(pd.unique(y_true)).issubset(set(loaded_classes)):
        raise ValueError(
            f"'CH' contains labels outside trained classes {loaded_classes.tolist()}."
        )

    plot_roc_auc(
        loaded_model,
        X,
        y_true,
        loaded_classes,
        save_path="roc_auc.png",
    )

    accuracy = accuracy_score(y_true, y_pred_numeric)
    target_names = [CLASS_LABELS.get(cls, str(cls)) for cls in loaded_classes]
    report = classification_report(
        y_true,
        y_pred_numeric,
        labels=loaded_classes,
        target_names=target_names,
        zero_division=0,
    )
    print(f"\nAccuracy Score: {accuracy:.4f}\n")
    print("Classification Report:\n", report)

    with open("classification_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(
        y_true,
        y_pred_numeric,
        loaded_classes,
        save_path="confusion_matrix.png",
    )
    save_chrs_statistics(df, y_pred_numeric, output_file="chrs_statistics.txt")
    # ========================================================================


if __name__ == "__main__":
    main()
