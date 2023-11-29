import os, json
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics

from keras.models import load_model
from matplotlib import use as plt_use
from matplotlib import pyplot as plt

from ..dicom.dataframes import DefaultDataFrame
from ..mCNN.mCNN import MixedCNNTrainer

# Disable the GUI
plt_use('agg')


def find_subdirs_starting_with(parent_dir, substring):
    subdirs = []
    for item in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, item)
        if os.path.isdir(full_path) and item.startswith(substring):
            subdirs.append(full_path)
    return sorted(subdirs)


def normalize_png(path):
    """
    Make sure we have a nice .png at the end.
    """
    png_split = path.rpartition(".png")
    png_path = png_split[0] + png_split[1] if png_split[0] else png_split[-1] + ".png"
    return png_path


def classification_report(targets, predictions_bool):
    return skmetrics.classification_report(targets, predictions_bool)


def confusion_matrix(targets, predictions_bool):
    return skmetrics.confusion_matrix(targets, predictions_bool)


def roc_auc_stats(targets, predictions_raw):
    """
    Return a tuple of the fpr, tpr, thresholds, and auc
    """
    fpr, tpr, thresholds  = skmetrics.roc_curve(targets, predictions_raw)
    auc = skmetrics.roc_auc_score(targets, predictions_raw)
    return fpr, tpr, thresholds, auc


def precision_recall_fscore_support(targets, predictions_bool, **kwargs):
    """
    Returns (precision, recall, fscore, support)
    """
    return skmetrics.precision_recall_fscore_support(targets, predictions_bool, **kwargs)


def get_precision_recall_arrays(targets, predictions_raw):
    """
    Return a tuple of the (precision, recall, threshold) arrays for plotting
    """
    return skmetrics.precision_recall_curve(targets, predictions_raw)


def simple_accuracy(targets, predictions_bool):
    """
    Plain accuracy
    Returns float
    """
    return skmetrics.accuracy_score(targets, predictions_bool)


def balanced_accuracy(targets, predictions_bool):
    """
    Better than regular accuracy for skewed datasets
    Returns float
    """
    return skmetrics.balanced_accuracy_score(targets, predictions_bool)


def detection_error_tradeoff(targets, predictions_raw):
    """
    DET
    Returns tuple (FPR, FNR, Thresholds)
    """
    return skmetrics.det_curve(targets, predictions_raw)


def create_confusion_matrix_plot(targets, predictions_bool, fig=None, ax=None, im_kw=None, text_kw=None):
    """
    Get a confusion matrix plot
    sklearn recommends using the from_predictions utility

    Return the fig and ax
    """
    if not all([fig, ax]):
        fig, ax = plt.subplots()
    skmetrics.ConfusionMatrixDisplay.from_predictions(
        targets, predictions_bool, display_labels=["Control", "At-risk"]
    ).plot(ax=ax, im_kw=im_kw, text_kw=text_kw)
    return fig, ax


def create_roc_plot(targets, predictions_raw, fig=None, ax=None, **kwargs):
    """
    Generate our ROC plot

    Return the fix and ax
    """
    if not all([fig, ax]):
        fig, ax = plt.subplots()
    skmetrics.RocCurveDisplay.from_predictions(
        targets, predictions_raw, **kwargs
    ).plot(ax=ax)
    return fig, ax


def create_precision_recall_plot(targets, predictions_bool, fig=None, ax=None, **kwargs):
    """
    Generate a P-R plot
    """
    if not all([fig, ax]):
        fig, ax = plt.subplots()
    skmetrics.PrecisionRecallDisplay.from_predictions(
        targets, predictions_bool, **kwargs
    ).plot(ax=ax)
    return fig, ax


def create_det_plot(targets, predictions_raw, fig=None, ax=None, **kwargs):
    if not all([fig, ax]):
        fig, ax = plt.subplots()
    skmetrics.DetCurveDisplay.from_predictions(
        targets, predictions_raw, **kwargs
    ).plot(ax=ax)
    return fig, ax


def get_trial_predictions_targets(trial_path):
    """
    Given a trial path, get the test targets and predictions for its saved model
    Returns tuple of (targets, predictions)
    """
    # Load the latest keras model
    checkpoints = find_subdirs_starting_with(trial_path, "checkpoint")
    if not checkpoints:
        raise RuntimeError(f"No checkpoints recorded (so no keras model saved) at path {trial_path}")
    model_path = os.path.join(checkpoints[-1], "model.keras")
    model = load_model(model_path)

    # Determine downscale factor and image reshape
    json_path = os.path.join(trial_path, "params.json")
    with open(json_path, 'r') as jf:
        params = json.load(jf)

    downscale = params["downscale_factor"]
    image_reshape = tuple(params["image_reshape"])

    # We will assume default dataset
    df = DefaultDataFrame()
    trainer = MixedCNNTrainer(df, downscale_factor=downscale, image_reshape=image_reshape)

    # If the Keras model input_shape has length 2, then this is Mixed mode, and it wants attrs
    # If so, we toggle keras_sequential off
    print(f"Model input shape: {model.input_shape}")
    keras_sequential = not bool(len(model.input_shape) == 2)
    trainer.populate_learning_data(keras_sequential=keras_sequential)

    if keras_sequential:
        predictions = model.predict(x=trainer.test_input)
    else:
        predictions = model.predict(x=[trainer.test_input, trainer.test_attrs])
    targets = trainer.test_targets
    print(f"Loaded test targets and predictions: {targets.shape} results")

    return targets, predictions.flatten()


def run_suite(trial_path, overwrite=False, threshold=0.5):
    """
    Run the stats suite on a trial path.
    If overwrite is false, throw exception if the directory already exists
    """
    run_name = trial_path.rpartition("/")[-1]
    short_name = run_name.split("_")[-1][:5]  # First 5 letters of ID
    print(f"\n\n<><><<><>< Calculating and writing stats to {run_name} ><><><><>")

    # See if the stats path exists
    stats_dir = os.path.join(trial_path, "stats")
    if not overwrite and os.path.exists(stats_dir):
        raise RuntimeError(f"Overwrite is off, and stats dir exists: {stats_dir}")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    try:
        targets, predictions_raw = get_trial_predictions_targets(trial_path)
    except RuntimeError as RE:
        print(RE)
        return
    predictions_bool = (predictions_raw > threshold).astype(float)

    # 1. Precision, Recall, F1, Support
    print("Calculating Precision, Recall, F1, Support")
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions_bool)
    precision_array, recall_array, _ = get_precision_recall_arrays(targets, predictions_raw)

    # 2. FPR, TPR, Thresholds, AUC
    print("Calculating FPR, TPR, Thresholds, AUC")
    fpr, tpr, thresholds, auc = roc_auc_stats(targets, predictions_raw)

    # 3. Confusion Matrix at 0.5
    print("Calculating Confusion Matrix")
    cm = confusion_matrix(targets, predictions_bool)

    # 4. Plain and Balanced Accuracy
    print("Calculating Simple and Balanced Accuracies")
    plain_acc = simple_accuracy(targets, predictions_bool)
    balanced_acc = balanced_accuracy(targets, predictions_bool)

    # 5. Detection Error Tradeoff
    print("Calculating Detection Error Tradeoff")
    fpr, fnr, _ = detection_error_tradeoff(targets, predictions_raw)

    # 6. Save these stats to JSON
    all_stats = {
        # "thresholds": thresholds.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "fnr": fnr.tolist(),
        "confusion_matrix": cm.tolist(),
        "precision": precision.tolist(),
        "precision_array": precision_array.tolist(),
        "recall": recall.tolist(),
        "recall_array": recall_array.tolist(),
        "simple_accuracy": plain_acc,
        "balanced_acc": balanced_acc,
        "f1": f1.tolist(),
        "support": support.tolist(),
        "targets": targets.tolist(),
        "predictions": predictions_raw.tolist()
    }
    json_path = os.path.join(stats_dir, "stats.json")
    with open(json_path, 'w') as jf:
        json.dump(all_stats, jf, indent=1)

    # Let's also save the classification report
    print("Writing classification report")
    report = classification_report(targets, predictions_bool)
    report_path = os.path.join(stats_dir, "class_report.txt")
    with open(report_path, 'w') as cf:
        cf.write(report)

    # Some plots
    # 7. ROC Curve
    print("Plotting ROC Curve")
    fig, ax = create_roc_plot(targets, predictions_raw, name=short_name)
    # ax.set_title(f"ROC [{run_name}]")
    fig.tight_layout()
    fig.savefig(stats_dir + "/roc.png")
    plt.close(fig)
    del fig, ax

    # 8. Confusion Matrix
    print("Plotting Confusion Matrix")
    fig, ax = create_confusion_matrix_plot(targets, predictions_bool)
    fig.tight_layout()
    # ax.set_title(f"Threshold = 0.5 [{run_name}]")
    fig.savefig(stats_dir + "/confusion.png")

    # 9. Precision/Recall
    print("Plotting Precision-Recall Curve")
    fig, ax = create_precision_recall_plot(targets, predictions_raw, name=short_name)
    fig.tight_layout()
    fig.savefig(stats_dir + "/precision-recall.png")

    # 10. DET Plot
    print("Plotting DET Curve")
    fig, ax = create_det_plot(targets, predictions_raw, name=short_name)
    fig.tight_layout()
    fig.savefig(stats_dir + "/det.png")

    # Print class report for fun
    print(report)
    print("\n")
