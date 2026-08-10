# Heart Disease Prediction Using K-Nearest Neighbors

# CAI 4105 - Emre Akilli

## Project Objective

The model predicts one of two outcomes:

- `0`: no heart disease
- `1`: heart disease is present

The project evaluates overall classification performance while giving special attention to false negatives. In this context, a false negative means that a patient who has heart disease is classified as not having it.

## Dataset

The project uses `heart_cleveland_upload.csv`, a preprocessed version of the Cleveland heart disease dataset. The supplied CSV contains:

- 297 patient records
- 13 clinical input features
- 1 binary target column
- No missing values
- 160 patients labeled `0`
- 137 patients labeled `1`

The file already has a binary target. It does not require conversion from the original target values of 0 through 4, and the patient records are not altered by the program.

## Feature Descriptions

The CSV retains its original column names:

- `age`: age in years
- `sex`: sex recorded as 1 for male and 0 for female
- `cp`: chest pain type
- `trestbps`: resting blood pressure in mm Hg
- `chol`: serum cholesterol in mg/dL
- `fbs`: whether fasting blood sugar is above 120 mg/dL
- `restecg`: resting electrocardiographic result
- `thalach`: maximum heart rate achieved
- `exang`: whether exercise-induced angina is present
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: slope of the peak exercise ST segment
- `ca`: number of major vessels colored by fluoroscopy
- `thal`: thalassemia category
- `condition`: binary classification target, where 0 means no heart disease and 1 means heart disease is present

Although several categorical features are stored as numbers, their values represent categories rather than continuous measurements.

## Project Structure

```text
heart-disease-knn/
├── data/
│   └── heart_cleveland_upload.csv
├── src/
│   └── heart_disease_knn.py
├── results/
├── requirements.txt
└── README.md
```

The `results` folder is included in the project and is also created automatically by the program if it is missing.

## Requirements

This project requires Python 3.10 or newer and the following Python libraries:

- pandas
- NumPy
- Matplotlib
- scikit-learn

The minimum package versions are listed in `requirements.txt`.

## Installation

From the `heart-disease-knn` folder, create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Windows Command Prompt, use:

```bat
.venv\Scripts\activate.bat
```

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Setup

Place `heart_cleveland_upload.csv` inside the `data` folder. The expected path is:

```text
heart-disease-knn/data/heart_cleveland_upload.csv
```

The file name should not be changed unless the dataset path in the Python code is updated as well. The script builds this path from its own location, so it works in VS Code on Windows even when the terminal starts in a different working directory.

## Running the Program

From the project folder, run:

```powershell
python src/heart_disease_knn.py
```

The program validates and summarizes the dataset, evaluates the candidate KNN configurations and probability thresholds, trains the selected model, reports test performance, and saves three plots.

## Preprocessing

The features are handled according to their meaning:

- Continuous features use mean imputation as a safeguard and Min-Max scaling.
- Binary features use most-frequent imputation and remain numeric.
- Categorical features use most-frequent imputation and one-hot encoding.

The provided CSV has no missing values, so the imputers do not replace any values in the current data. They remain in the workflow as safeguards for an unexpected missing feature value.

All preprocessing is kept inside scikit-learn pipelines. It is learned from the training data during each fit rather than from the full dataset, which prevents information from the testing or validation records from leaking into model training. Min-Max scaling is particularly important for KNN because the method compares distances, and unscaled features with larger ranges could have too much influence.

The `ca` feature records a vessel count, but it can also be read as a small set of category values. The model search therefore compares two controlled treatments. One strategy one-hot encodes `ca` as categorical, while the other imputes it as an ordered numerical count and applies Min-Max scaling. The remaining feature groups stay unchanged.

## Model Training

The data is divided with an 80/20 stratified train-test split. Stratification keeps the target class proportions similar in the training and testing sets. The random state is fixed at 42 so that the split can be reproduced.

The training set is used to evaluate neighbor counts of 1, 3, 5, 7, 9, 11, 13, and 15. Each candidate is evaluated with 10-fold stratified cross-validation. The testing set is kept untouched during this stage.

## Expanded Model and Threshold Search

The model search now compares more than the number of neighbors. It evaluates:

- Uniform and distance-based voting
- Manhattan and Euclidean distance
- Categorical and ordered-numerical handling of `ca`
- Probability thresholds from 0.30 through 0.60

The configuration search first finds the best mean cross-validation accuracy. Models remain eligible when their accuracy is no more than 0.02 below that value. Recall is then prioritized among the eligible models, followed by F1 score, accuracy, the smaller neighbor count, and distance weighting. This rule aims to reduce false negatives without accepting a major loss in general classification performance.

After selecting a model configuration, the program generates out-of-fold probabilities for the training records with `cross_val_predict`. It compares probability thresholds using those predictions and applies the same 0.02 accuracy tolerance before choosing the highest-recall eligible threshold. Lowering a threshold can identify more positive cases and reduce false negatives, but it may also increase false positives and lower precision. The threshold search makes that trade-off visible instead of assuming that 0.50 is always best.

All configuration and threshold tuning takes place on the training data through cross-validation. The test set is not examined during either search. It is used once, after all choices have been made, to evaluate the final fitted pipeline with the selected threshold.

## Evaluation Metrics

The program reports:

- **Accuracy**, the proportion of all predictions that are correct
- **Precision**, the proportion of predicted heart disease cases that are correct
- **Recall**, the proportion of actual heart disease cases identified by the model
- **F1 score**, a balance between precision and recall
- **Confusion matrix**, a count of true negatives, false positives, false negatives, and true positives

Recall receives priority during model selection because a false negative occurs when a patient who has heart disease is incorrectly classified as not having it. The other metrics are still reported so that the final result is not judged from a single measure.

## Generated Results

Running the program creates:

- `results/knn_neighbor_comparison.png`, which summarizes how neighbor count affects the four mean cross-validation metrics across the tested configurations
- `results/threshold_comparison.png`, which compares accuracy, precision, recall, and F1 score across the tested probability thresholds
- `results/confusion_matrix.png`, which shows the final model's predictions on the untouched test set

The exact scores, model configuration, and selected threshold are printed when the program runs. No improved score is assumed in advance.

## Limitations

The dataset is relatively small and represents a limited patient population. Performance on these records may not generalize to patients from other regions, age groups, or clinical settings.

KNN is sensitive to feature scaling and to the chosen number of neighbors. The pipeline addresses scaling consistently, but model performance can still vary with the data split and the candidate values considered. More importantly, a machine learning classification result does not establish a medical diagnosis.

## Educational Disclaimer

This model is for educational purposes only. It is not a substitute for diagnosis, advice, or treatment from a qualified healthcare professional.
