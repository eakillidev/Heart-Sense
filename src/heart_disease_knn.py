"""Train and evaluate a KNN classifier on the Cleveland heart disease data."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "heart_cleveland_upload.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
NEIGHBOR_COMPARISON_PLOT_PATH = RESULTS_DIRECTORY / "knn_neighbor_comparison.png"
THRESHOLD_COMPARISON_PLOT_PATH = RESULTS_DIRECTORY / "threshold_comparison.png"
CONFUSION_MATRIX_PLOT_PATH = RESULTS_DIRECTORY / "confusion_matrix.png"

TARGET_COLUMN_NAME = "condition"
CONTINUOUS_FEATURE_NAMES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURE_NAMES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURE_NAMES = ["cp", "restecg", "slope", "thal"]
EXPECTED_COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "condition",
]
FEATURE_DESCRIPTIONS = {
    "age": "age in years",
    "sex": "1 represents male and 0 represents female",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure in mm Hg",
    "chol": "serum cholesterol in mg/dL",
    "fbs": "fasting blood sugar above 120 mg/dL",
    "restecg": "resting electrocardiographic result",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "slope of the peak exercise ST segment",
    "ca": "number of major vessels colored by fluoroscopy",
    "thal": "thalassemia category",
    "condition": "binary heart disease classification",
}
NEIGHBOR_COUNTS = [1, 3, 5, 7, 9, 11, 13, 15]
VOTING_METHODS = ["uniform", "distance"]
DISTANCE_POWERS = [1, 2]
CA_HANDLING_STRATEGIES = ["ca_categorical", "ca_numerical"]
PROBABILITY_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MAXIMUM_ACCEPTABLE_ACCURACY_DROP = 0.02
MAXIMUM_ACCEPTABLE_THRESHOLD_ACCURACY_DROP = 0.02

PREVIOUS_TEST_ACCURACY = 0.8833
PREVIOUS_TEST_RECALL = 0.7500
PREVIOUS_FALSE_NEGATIVES = 7


def load_heart_disease_data() -> pd.DataFrame:
    """Load the supplied heart disease CSV from the project's data folder."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "The heart disease dataset was not found. Place the original file at "
            f"'{DATASET_PATH}' and keep the file name heart_cleveland_upload.csv."
        )

    try:
        heart_disease_data = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"The dataset at '{DATASET_PATH}' is empty. Replace it with the provided "
            "heart_cleveland_upload.csv file."
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"The dataset at '{DATASET_PATH}' could not be parsed as a CSV file. "
            "Check that the supplied file has not been changed."
        ) from error

    return heart_disease_data


def validate_heart_disease_data(heart_disease_data: pd.DataFrame) -> None:
    """Validate the dataset schema, target values, and feature value types."""
    if heart_disease_data.empty:
        raise ValueError(
            "The dataset contains no patient records. Restore the provided "
            "heart_cleveland_upload.csv file in the data folder."
        )

    actual_column_count = len(heart_disease_data.columns)
    if actual_column_count != 14:
        raise ValueError(
            "The dataset must contain exactly 14 columns, but "
            f"{actual_column_count} were found. Use the provided preprocessed CSV."
        )

    missing_column_names = [
        column_name
        for column_name in EXPECTED_COLUMN_NAMES
        if column_name not in heart_disease_data.columns
    ]
    if missing_column_names:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{', '.join(missing_column_names)}. Restore the original header row."
        )

    unexpected_column_names = [
        column_name
        for column_name in heart_disease_data.columns
        if column_name not in EXPECTED_COLUMN_NAMES
    ]
    if unexpected_column_names:
        raise ValueError(
            "The dataset contains unexpected columns: "
            f"{', '.join(unexpected_column_names)}. Use the supplied CSV unchanged."
        )

    if heart_disease_data[TARGET_COLUMN_NAME].isna().any():
        raise ValueError(
            "The condition target contains missing values. Target labels cannot be "
            "imputed; restore the provided CSV before training the model."
        )

    target_values = set(heart_disease_data[TARGET_COLUMN_NAME].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            "The condition column must contain only binary values 0 and 1, but "
            f"these values were found: {sorted(target_values, key=str)}. "
            "Do not use the unprocessed 0-through-4 target."
        )

    feature_names = [
        column_name
        for column_name in EXPECTED_COLUMN_NAMES
        if column_name != TARGET_COLUMN_NAME
    ]
    non_numeric_feature_names: list[str] = []
    for feature_name in feature_names:
        feature_values = heart_disease_data[feature_name]
        converted_feature_values = pd.to_numeric(feature_values, errors="coerce")
        contains_non_numeric_values = (
            converted_feature_values.isna() & feature_values.notna()
        ).any()
        if contains_non_numeric_values:
            non_numeric_feature_names.append(feature_name)

    if non_numeric_feature_names:
        raise ValueError(
            "All input features must be numeric. Non-numeric values were found in: "
            f"{', '.join(non_numeric_feature_names)}. Correct the CSV using the "
            "original supplied data."
        )

    feature_missing_value_count = int(
        heart_disease_data[feature_names].isna().sum().sum()
    )
    if feature_missing_value_count > 0:
        warnings.warn(
            f"The dataset contains {feature_missing_value_count} unexpected missing "
            "feature value(s). The original CSV will not be modified; the "
            "preprocessing pipeline will impute these values using training data.",
            RuntimeWarning,
            stacklevel=2,
        )


def inspect_heart_disease_data(heart_disease_data: pd.DataFrame) -> None:
    """Print a readable summary of the dataset and selected statistics."""
    input_feature_count = len(heart_disease_data.columns) - 1

    print("\nDataset Summary")
    print("---------------")
    print(f"Number of patient records: {len(heart_disease_data)}")
    print(f"Number of input features: {input_feature_count}")
    print(f"Number of total columns: {len(heart_disease_data.columns)}")
    print(f"Column names: {', '.join(heart_disease_data.columns)}")

    missing_values_by_column = heart_disease_data.isna().sum()
    print("\nMissing Values")
    print("--------------")
    for column_name, missing_value_count in missing_values_by_column.items():
        print(f"{column_name}: {missing_value_count}")
    print(f"Total number of missing values: {int(missing_values_by_column.sum())}")

    target_counts = (
        heart_disease_data[TARGET_COLUMN_NAME]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    total_record_count = len(heart_disease_data)
    print("\nTarget Distribution")
    print("-------------------")
    for target_value in (0, 1):
        target_count = int(target_counts.loc[target_value])
        target_percentage = (target_count / total_record_count) * 100
        print(
            f"Condition {target_value}: {target_count} records "
            f"({target_percentage:.2f}%)"
        )

    statistics_feature_names = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    descriptive_statistics = heart_disease_data[
        statistics_feature_names
    ].describe()
    print("\nDescriptive Statistics")
    print("----------------------")
    print(descriptive_statistics.round(2).to_string())


def create_preprocessing_transformer(
    ca_handling_strategy: str,
) -> ColumnTransformer:
    """Create preprocessing for the requested treatment of the ca feature."""
    if ca_handling_strategy not in CA_HANDLING_STRATEGIES:
        raise ValueError(
            f"Unknown ca handling strategy '{ca_handling_strategy}'. Choose one of: "
            f"{', '.join(CA_HANDLING_STRATEGIES)}."
        )

    continuous_feature_pipeline = Pipeline(
        steps=[
            ("mean_imputer", SimpleImputer(strategy="mean")),
            # Scaling prevents larger measurement ranges from dominating distances.
            ("min_max_scaler", MinMaxScaler()),
        ]
    )
    binary_feature_pipeline = Pipeline(
        steps=[("most_frequent_imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_feature_pipeline = Pipeline(
        steps=[
            ("most_frequent_imputer", SimpleImputer(strategy="most_frequent")),
            # Numeric category codes are labels, not continuous measurements.
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    categorical_feature_names = CATEGORICAL_FEATURE_NAMES.copy()
    if ca_handling_strategy == "ca_categorical":
        categorical_feature_names.append("ca")

    preprocessing_steps: list[tuple[str, Pipeline, list[str]]] = [
        (
            "continuous_features",
            continuous_feature_pipeline,
            CONTINUOUS_FEATURE_NAMES,
        ),
        ("binary_features", binary_feature_pipeline, BINARY_FEATURE_NAMES),
        (
            "categorical_features",
            categorical_feature_pipeline,
            categorical_feature_names,
        ),
    ]

    # ca is tested both as category labels and as an ordered vessel count because
    # either interpretation is defensible; cross-validation decides between them.
    if ca_handling_strategy == "ca_numerical":
        ca_numerical_pipeline = Pipeline(
            steps=[
                (
                    "most_frequent_imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                ("min_max_scaler", MinMaxScaler()),
            ]
        )
        preprocessing_steps.append(
            ("ca_numerical_feature", ca_numerical_pipeline, ["ca"])
        )

    preprocessing_transformer = ColumnTransformer(
        transformers=preprocessing_steps
    )
    return preprocessing_transformer


def create_stratified_cross_validation() -> StratifiedKFold:
    """Create the shared reproducible ten-fold validation strategy."""
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def create_knn_model_pipeline(
    neighbor_count: int,
    voting_method: str,
    distance_power: int,
    ca_handling_strategy: str,
) -> Pipeline:
    """Create one complete preprocessing and KNN model pipeline."""
    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                create_preprocessing_transformer(ca_handling_strategy),
            ),
            (
                "knn_classifier",
                KNeighborsClassifier(
                    n_neighbors=neighbor_count,
                    weights=voting_method,
                    p=distance_power,
                ),
            ),
        ]
    )
    return model_pipeline


def evaluate_knn_configurations(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
) -> pd.DataFrame:
    """Evaluate all requested KNN configurations on training folds only."""
    stratified_cross_validation = create_stratified_cross_validation()
    scoring_metrics = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    evaluation_rows: list[dict[str, float | int | str]] = []

    for neighbor_count in NEIGHBOR_COUNTS:
        for voting_method in VOTING_METHODS:
            for distance_power in DISTANCE_POWERS:
                for ca_handling_strategy in CA_HANDLING_STRATEGIES:
                    model_pipeline = create_knn_model_pipeline(
                        neighbor_count,
                        voting_method,
                        distance_power,
                        ca_handling_strategy,
                    )
                    # Every preprocessing fit occurs inside a training fold. The
                    # final test records are not used to compare configurations.
                    cross_validation_scores = cross_validate(
                        estimator=model_pipeline,
                        X=training_features,
                        y=training_labels,
                        cv=stratified_cross_validation,
                        scoring=scoring_metrics,
                    )
                    evaluation_rows.append(
                        {
                            "neighbor_count": neighbor_count,
                            "weights": voting_method,
                            "distance_power": distance_power,
                            "distance_metric": (
                                "Manhattan (p=1)"
                                if distance_power == 1
                                else "Euclidean (p=2)"
                            ),
                            "ca_handling_strategy": ca_handling_strategy,
                            "mean_accuracy": float(
                                np.mean(
                                    cross_validation_scores["test_accuracy"]
                                )
                            ),
                            "mean_precision": float(
                                np.mean(
                                    cross_validation_scores["test_precision"]
                                )
                            ),
                            "mean_recall": float(
                                np.mean(cross_validation_scores["test_recall"])
                            ),
                            "mean_f1": float(
                                np.mean(cross_validation_scores["test_f1"])
                            ),
                        }
                    )

    configuration_evaluation_results = pd.DataFrame(evaluation_rows)
    highest_mean_accuracy = float(
        configuration_evaluation_results["mean_accuracy"].max()
    )
    minimum_eligible_accuracy = (
        highest_mean_accuracy - MAXIMUM_ACCEPTABLE_ACCURACY_DROP
    )
    configuration_evaluation_results["accuracy_eligible"] = (
        configuration_evaluation_results["mean_accuracy"]
        >= minimum_eligible_accuracy
    )
    return configuration_evaluation_results


def select_best_knn_configuration(
    configuration_evaluation_results: pd.DataFrame,
) -> pd.Series:
    """Select the highest-recall model inside the accuracy tolerance."""
    eligible_configuration_results = configuration_evaluation_results.loc[
        configuration_evaluation_results["accuracy_eligible"]
    ].copy()
    if eligible_configuration_results.empty:
        raise ValueError(
            "No KNN configuration met the cross-validation accuracy tolerance. "
            "Check the eligibility calculation and configured tolerance."
        )

    eligible_configuration_results["distance_weighting_preference"] = (
        eligible_configuration_results["weights"] == "distance"
    ).astype(int)
    # Recall is prioritized, while the accuracy tolerance prevents accepting a
    # large loss in overall validation performance merely to raise recall.
    ranked_configuration_results = eligible_configuration_results.sort_values(
        by=[
            "mean_recall",
            "mean_f1",
            "mean_accuracy",
            "neighbor_count",
            "distance_weighting_preference",
        ],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    )
    return ranked_configuration_results.iloc[0]


def generate_cross_validated_probabilities(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    selected_configuration: pd.Series,
) -> np.ndarray:
    """Generate positive-class out-of-fold probabilities from training data."""
    selected_model_pipeline = create_knn_model_pipeline(
        int(selected_configuration["neighbor_count"]),
        str(selected_configuration["weights"]),
        int(selected_configuration["distance_power"]),
        str(selected_configuration["ca_handling_strategy"]),
    )
    # Out-of-fold probabilities let threshold selection use every training record
    # without predicting any record from a model fitted on that same record.
    # The test set is deliberately absent so it remains untouched until evaluation.
    cross_validated_probability_matrix = cross_val_predict(
        estimator=selected_model_pipeline,
        X=training_features,
        y=training_labels,
        cv=create_stratified_cross_validation(),
        method="predict_proba",
    )
    if cross_validated_probability_matrix.shape[1] != 2:
        raise ValueError(
            "Threshold tuning expected two probability columns for target classes "
            f"0 and 1, but found shape {cross_validated_probability_matrix.shape}."
        )
    return cross_validated_probability_matrix[:, 1]


def evaluate_probability_thresholds(
    training_labels: pd.Series,
    positive_class_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate candidate thresholds using out-of-fold training probabilities."""
    threshold_evaluation_rows: list[dict[str, float | int]] = []

    for probability_threshold in PROBABILITY_THRESHOLDS:
        threshold_predicted_labels = (
            positive_class_probabilities >= probability_threshold
        ).astype(int)
        confusion_matrix_values = confusion_matrix(
            training_labels,
            threshold_predicted_labels,
            labels=[0, 1],
        )
        if confusion_matrix_values.shape != (2, 2):
            raise ValueError(
                "Threshold validation requires a 2 by 2 confusion matrix, but "
                f"found shape {confusion_matrix_values.shape}."
            )
        (
            _,
            false_positives,
            false_negatives,
            _,
        ) = confusion_matrix_values.ravel()
        threshold_evaluation_rows.append(
            {
                "threshold": probability_threshold,
                "accuracy": accuracy_score(
                    training_labels, threshold_predicted_labels
                ),
                "precision": precision_score(
                    training_labels,
                    threshold_predicted_labels,
                    zero_division=0,
                ),
                "recall": recall_score(
                    training_labels,
                    threshold_predicted_labels,
                    zero_division=0,
                ),
                "f1": f1_score(
                    training_labels,
                    threshold_predicted_labels,
                    zero_division=0,
                ),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
            }
        )

    threshold_evaluation_results = pd.DataFrame(threshold_evaluation_rows)
    highest_threshold_accuracy = float(
        threshold_evaluation_results["accuracy"].max()
    )
    minimum_eligible_threshold_accuracy = (
        highest_threshold_accuracy
        - MAXIMUM_ACCEPTABLE_THRESHOLD_ACCURACY_DROP
    )
    threshold_evaluation_results["accuracy_eligible"] = (
        threshold_evaluation_results["accuracy"]
        >= minimum_eligible_threshold_accuracy
    )
    return threshold_evaluation_results


def select_best_probability_threshold(
    threshold_evaluation_results: pd.DataFrame,
) -> pd.Series:
    """Select a high-recall threshold inside the validation accuracy tolerance."""
    eligible_threshold_results = threshold_evaluation_results.loc[
        threshold_evaluation_results["accuracy_eligible"]
    ].copy()
    if eligible_threshold_results.empty:
        raise ValueError(
            "No probability threshold met the validation accuracy tolerance. "
            "Check the eligibility calculation and configured tolerance."
        )

    eligible_threshold_results["distance_from_standard_threshold"] = (
        eligible_threshold_results["threshold"] - 0.50
    ).abs()
    ranked_threshold_results = eligible_threshold_results.sort_values(
        by=[
            "recall",
            "f1",
            "precision",
            "distance_from_standard_threshold",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked_threshold_results.iloc[0]


def train_final_knn_model(
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    selected_configuration: pd.Series,
) -> Pipeline:
    """Fit the final preprocessing and KNN pipeline on all training records."""
    model_pipeline = create_knn_model_pipeline(
        int(selected_configuration["neighbor_count"]),
        str(selected_configuration["weights"]),
        int(selected_configuration["distance_power"]),
        str(selected_configuration["ca_handling_strategy"]),
    )
    model_pipeline.fit(training_features, training_labels)
    return model_pipeline


def evaluate_final_knn_model(
    model_pipeline: Pipeline,
    testing_features: pd.DataFrame,
    testing_labels: pd.Series,
    selected_probability_threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evaluate one thresholded prediction on the untouched testing records."""
    predicted_probability_matrix = model_pipeline.predict_proba(testing_features)
    if predicted_probability_matrix.shape[1] != 2:
        raise ValueError(
            "Final evaluation expected two probability columns for target classes "
            f"0 and 1, but found shape {predicted_probability_matrix.shape}."
        )
    predicted_probabilities = predicted_probability_matrix[:, 1]
    predicted_labels = (
        predicted_probabilities >= selected_probability_threshold
    ).astype(int)
    test_accuracy = accuracy_score(testing_labels, predicted_labels)
    test_precision = precision_score(
        testing_labels, predicted_labels, zero_division=0
    )
    test_recall = recall_score(testing_labels, predicted_labels, zero_division=0)
    test_f1_score = f1_score(testing_labels, predicted_labels, zero_division=0)
    confusion_matrix_values = confusion_matrix(
        testing_labels,
        predicted_labels,
        labels=[0, 1],
    )

    if confusion_matrix_values.shape != (2, 2):
        raise ValueError(
            "The confusion matrix must be 2 by 2 for binary classification, but "
            f"its shape is {confusion_matrix_values.shape}. Check the target labels "
            "and model predictions."
        )

    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix_values.ravel()
    )
    readable_classification_report = classification_report(
        testing_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["No Heart Disease", "Heart Disease"],
        zero_division=0,
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 score: {test_f1_score:.4f}")
    print("\nClassification report:")
    print(readable_classification_report)
    print("Confusion matrix:")
    print(confusion_matrix_values)
    print(f"True negatives: {true_negatives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"True positives: {true_positives}")

    final_test_metrics: dict[str, float | int] = {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1_score),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
    }
    return confusion_matrix_values, final_test_metrics


def print_knn_configuration_results(
    configuration_evaluation_results: pd.DataFrame,
) -> None:
    """Print all KNN model-search configurations in a readable table."""
    display_table = configuration_evaluation_results[
        [
            "neighbor_count",
            "weights",
            "distance_metric",
            "ca_handling_strategy",
            "mean_accuracy",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "accuracy_eligible",
        ]
    ].rename(
        columns={
            "neighbor_count": "Neighbors",
            "weights": "Weights",
            "distance_metric": "Distance Metric",
            "ca_handling_strategy": "ca Strategy",
            "mean_accuracy": "Mean Accuracy",
            "mean_precision": "Mean Precision",
            "mean_recall": "Mean Recall",
            "mean_f1": "Mean F1",
            "accuracy_eligible": "Accuracy Eligible",
        }
    )
    for metric_column in [
        "Mean Accuracy",
        "Mean Precision",
        "Mean Recall",
        "Mean F1",
    ]:
        display_table[metric_column] = display_table[metric_column].map(
            lambda metric_value: f"{metric_value:.4f}"
        )
    display_table["Accuracy Eligible"] = display_table[
        "Accuracy Eligible"
    ].map({True: "Yes", False: "No"})

    print("\nKNN Configuration Cross-Validation Results")
    print("------------------------------------------")
    print(display_table.to_string(index=False))


def print_threshold_results(
    threshold_evaluation_results: pd.DataFrame,
) -> None:
    """Print the out-of-fold probability-threshold results."""
    display_table = threshold_evaluation_results.rename(
        columns={
            "threshold": "Threshold",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "false_positives": "False Positives",
            "false_negatives": "False Negatives",
            "accuracy_eligible": "Accuracy Eligible",
        }
    ).copy()
    for metric_column in [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]:
        display_table[metric_column] = display_table[metric_column].map(
            lambda metric_value: f"{metric_value:.4f}"
        )
    display_table["Accuracy Eligible"] = display_table[
        "Accuracy Eligible"
    ].map({True: "Yes", False: "No"})

    print("\nProbability Threshold Cross-Validation Results")
    print("----------------------------------------------")
    print(display_table.to_string(index=False))


def save_knn_configuration_plot(
    configuration_evaluation_results: pd.DataFrame,
) -> None:
    """Save average configuration performance for each neighbor count."""
    neighbor_performance_summary = (
        configuration_evaluation_results.groupby("neighbor_count", as_index=False)[
            ["mean_accuracy", "mean_precision", "mean_recall", "mean_f1"]
        ]
        .mean()
        .sort_values("neighbor_count")
    )
    figure, plot_axes = plt.subplots(figsize=(9, 6))
    metric_columns_and_labels = {
        "mean_accuracy": "Mean Accuracy",
        "mean_precision": "Mean Precision",
        "mean_recall": "Mean Recall",
        "mean_f1": "Mean F1 Score",
    }
    neighbor_count_values = neighbor_performance_summary["neighbor_count"]

    for metric_column, metric_label in metric_columns_and_labels.items():
        plot_axes.plot(
            neighbor_count_values,
            neighbor_performance_summary[metric_column],
            marker="o",
            linewidth=2,
            label=metric_label,
        )

    plot_axes.set_title(
        "Average KNN Cross-Validation Performance by Neighbor Count"
    )
    plot_axes.set_xlabel("Number of Neighbors")
    plot_axes.set_ylabel("Average Mean Cross-Validation Score")
    plot_axes.set_xticks(neighbor_count_values)
    plot_axes.set_ylim(0.0, 1.05)
    plot_axes.grid(alpha=0.3)
    plot_axes.legend()
    figure.tight_layout()
    figure.savefig(NEIGHBOR_COMPARISON_PLOT_PATH, dpi=200)
    plt.close(figure)


def save_threshold_comparison_plot(
    threshold_evaluation_results: pd.DataFrame,
    selected_probability_threshold: float,
) -> None:
    """Save threshold metrics and mark the selected probability threshold."""
    figure, plot_axes = plt.subplots(figsize=(9, 6))
    metric_columns_and_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
    }
    threshold_values = threshold_evaluation_results["threshold"]

    for metric_column, metric_label in metric_columns_and_labels.items():
        plot_axes.plot(
            threshold_values,
            threshold_evaluation_results[metric_column],
            marker="o",
            linewidth=2,
            label=metric_label,
        )

    plot_axes.axvline(
        selected_probability_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Selected Threshold ({selected_probability_threshold:.2f})",
    )
    plot_axes.set_title(
        "Out-of-Fold Training Performance by Probability Threshold"
    )
    plot_axes.set_xlabel("Probability Threshold")
    plot_axes.set_ylabel("Cross-Validated Score")
    plot_axes.set_xticks(threshold_values)
    plot_axes.set_ylim(0.0, 1.05)
    plot_axes.grid(alpha=0.3)
    plot_axes.legend()
    figure.tight_layout()
    figure.savefig(THRESHOLD_COMPARISON_PLOT_PATH, dpi=200)
    plt.close(figure)


def save_confusion_matrix_plot(confusion_matrix_values: np.ndarray) -> None:
    """Save a labeled image of the binary confusion matrix."""
    if confusion_matrix_values.shape != (2, 2):
        raise ValueError(
            "The confusion matrix plot requires a 2 by 2 matrix, but "
            f"its shape is {confusion_matrix_values.shape}. Check the evaluation "
            "labels before plotting."
        )

    class_labels = ["No Heart Disease", "Heart Disease"]
    figure, plot_axes = plt.subplots(figsize=(7, 6))
    matrix_image = plot_axes.imshow(confusion_matrix_values, cmap="Blues")
    figure.colorbar(matrix_image, ax=plot_axes)
    plot_axes.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted Label",
        ylabel="Actual Label",
        title="KNN Confusion Matrix on the Test Set",
    )

    cell_text_threshold = confusion_matrix_values.max() / 2
    for row_index in range(2):
        for column_index in range(2):
            cell_value = int(confusion_matrix_values[row_index, column_index])
            plot_axes.text(
                column_index,
                row_index,
                str(cell_value),
                ha="center",
                va="center",
                color="white" if cell_value > cell_text_threshold else "black",
                fontsize=13,
                fontweight="bold",
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=200)
    plt.close(figure)


def ensure_results_directory() -> None:
    """Create the results directory or raise a clear file-system error."""
    try:
        RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Could not create the results folder at '{RESULTS_DIRECTORY}'. "
            "Check that the project folder is writable and try again."
        ) from error


def format_target_distribution(target_labels: pd.Series) -> str:
    """Format counts for target values 0 and 1 on one readable line."""
    target_counts = target_labels.value_counts().reindex([0, 1], fill_value=0)
    return (
        f"condition 0: {int(target_counts.loc[0])}, "
        f"condition 1: {int(target_counts.loc[1])}"
    )


def print_previous_result_comparison(
    final_test_metrics: dict[str, float | int],
) -> None:
    """Compare the one-time final result with the supplied previous result."""
    accuracy_change = (
        float(final_test_metrics["accuracy"]) - PREVIOUS_TEST_ACCURACY
    )
    recall_change = float(final_test_metrics["recall"]) - PREVIOUS_TEST_RECALL
    false_negative_change = (
        int(final_test_metrics["false_negatives"]) - PREVIOUS_FALSE_NEGATIVES
    )

    print("\nComparison with Previous Final Result")
    print("-------------------------------------")
    print(f"Previous accuracy: {PREVIOUS_TEST_ACCURACY:.4f}")
    print(f"New accuracy: {float(final_test_metrics['accuracy']):.4f}")
    print(f"Accuracy change: {accuracy_change:+.4f}")
    print(f"Previous recall: {PREVIOUS_TEST_RECALL:.4f}")
    print(f"New recall: {float(final_test_metrics['recall']):.4f}")
    print(f"Recall change: {recall_change:+.4f}")
    print(f"Previous false negatives: {PREVIOUS_FALSE_NEGATIVES}")
    print(
        "New false negatives: "
        f"{int(final_test_metrics['false_negatives'])}"
    )
    print(f"False-negative count change: {false_negative_change:+d}")
    if recall_change > 0:
        print("Outcome: Recall improved on the one-time final test evaluation.")
    else:
        print(
            "Outcome: Recall did not improve on the final test evaluation. "
            "The test result was not used to select another model."
        )


def main() -> None:
    """Run data inspection, model selection, final evaluation, and plotting."""
    heart_disease_data = load_heart_disease_data()
    validate_heart_disease_data(heart_disease_data)
    inspect_heart_disease_data(heart_disease_data)
    ensure_results_directory()

    feature_data = heart_disease_data.drop(columns=TARGET_COLUMN_NAME)
    target_labels = heart_disease_data[TARGET_COLUMN_NAME]
    (
        training_features,
        testing_features,
        training_labels,
        testing_labels,
    ) = train_test_split(
        feature_data,
        target_labels,
        test_size=0.20,
        random_state=42,
        stratify=target_labels,
    )

    print("\nTraining and Testing Split")
    print("--------------------------")
    print(f"Number of training records: {len(training_features)}")
    print(f"Number of testing records: {len(testing_features)}")
    print(
        "Training target distribution: "
        f"{format_target_distribution(training_labels)}"
    )
    print(
        "Testing target distribution: "
        f"{format_target_distribution(testing_labels)}"
    )

    configuration_evaluation_results = evaluate_knn_configurations(
        training_features,
        training_labels,
    )
    print_knn_configuration_results(configuration_evaluation_results)
    save_knn_configuration_plot(configuration_evaluation_results)
    selected_configuration = select_best_knn_configuration(
        configuration_evaluation_results
    )

    print("\nSelected KNN Configuration")
    print("--------------------------")
    print(
        "Number of neighbors: "
        f"{int(selected_configuration['neighbor_count'])}"
    )
    print(f"Weights: {selected_configuration['weights']}")
    print(f"Distance metric: {selected_configuration['distance_metric']}")
    print(
        "ca handling strategy: "
        f"{selected_configuration['ca_handling_strategy']}"
    )
    print(
        "Mean cross-validation accuracy: "
        f"{float(selected_configuration['mean_accuracy']):.4f}"
    )
    print(
        "Mean cross-validation recall: "
        f"{float(selected_configuration['mean_recall']):.4f}"
    )
    print(
        "Mean cross-validation F1 score: "
        f"{float(selected_configuration['mean_f1']):.4f}"
    )

    cross_validated_probabilities = generate_cross_validated_probabilities(
        training_features,
        training_labels,
        selected_configuration,
    )
    threshold_evaluation_results = evaluate_probability_thresholds(
        training_labels,
        cross_validated_probabilities,
    )
    print_threshold_results(threshold_evaluation_results)
    selected_threshold_result = select_best_probability_threshold(
        threshold_evaluation_results
    )
    selected_probability_threshold = float(
        selected_threshold_result["threshold"]
    )
    print("\nSelected Probability Threshold")
    print("------------------------------")
    print(f"Threshold: {selected_probability_threshold:.2f}")
    print(
        "Cross-validated accuracy: "
        f"{float(selected_threshold_result['accuracy']):.4f}"
    )
    print(
        "Cross-validated recall: "
        f"{float(selected_threshold_result['recall']):.4f}"
    )
    print(
        "Cross-validated F1 score: "
        f"{float(selected_threshold_result['f1']):.4f}"
    )
    save_threshold_comparison_plot(
        threshold_evaluation_results,
        selected_probability_threshold,
    )

    model_pipeline = train_final_knn_model(
        training_features,
        training_labels,
        selected_configuration,
    )
    confusion_matrix_values, final_test_metrics = evaluate_final_knn_model(
        model_pipeline,
        testing_features,
        testing_labels,
        selected_probability_threshold,
    )
    save_confusion_matrix_plot(confusion_matrix_values)
    print_previous_result_comparison(final_test_metrics)

    print("\nGenerated Result Files")
    print("----------------------")
    print(NEIGHBOR_COMPARISON_PLOT_PATH)
    print(THRESHOLD_COMPARISON_PLOT_PATH)
    print(CONFUSION_MATRIX_PLOT_PATH)


if __name__ == "__main__":
    main()
