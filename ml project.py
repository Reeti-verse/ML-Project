
import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance


# ============================================================
# 1. LOAD THE OFFLINE DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Final_Marks_Data (1).csv")

FEATURES = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

TARGET = "Final Exam Marks (out of 100)"

# Valid input limits for the GUI
INPUT_LIMITS = {
    "Attendance (%)": (0, 100),
    "Internal Test 1 (out of 40)": (0, 40),
    "Internal Test 2 (out of 40)": (0, 40),
    "Assignment Score (out of 10)": (0, 10),
    "Daily Study Hours": (0, 24)
}

if not os.path.isfile(CSV_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_FILE}\n\n"
        "Keep the CSV file in the same folder as PROJECT.py."
    )

df = pd.read_csv(CSV_FILE)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Validate required columns
required_columns = FEATURES + [TARGET]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert model columns to numeric values
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values with missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# A row without a target cannot be used for supervised training.
df.dropna(subset=[TARGET], inplace=True)
df.reset_index(drop=True, inplace=True)

if len(df) < 5:
    raise ValueError("Not enough valid records to train the model.")

# Exclude Student_ID and all other non-feature columns
X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nSTUDENT PERFORMANCE PREDICTION")
print("=" * 45)
print("Dataset loaded successfully.")
print(f"Number of records: {len(df)}")
print(f"Number of input features: {len(FEATURES)}")
print("\nDataset information:")
print(df[required_columns].describe())


# ============================================================
# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Split before learning imputation or outlier-correction limits.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Fill missing input values using training-set medians only.
medians = X_train.median().fillna(0)

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# ============================================================
# 3. DETECT OUTLIERS USING THE IQR METHOD
# ============================================================

# Learn IQR limits from training data, not from test data.
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1

lower_bounds = Q1 - 1.5 * IQR
upper_bounds = Q3 + 1.5 * IQR

# Count outliers in the training data before correction.
outlier_counts = (
    (X_train < lower_bounds) |
    (X_train > upper_bounds)
).sum()

print("\nOUTLIER DETECTION (IQR METHOD)")
print("-" * 45)

for col in FEATURES:
    print(f"{col}: {int(outlier_counts[col])} outliers detected")

# ============================================================
# 4. CORRECT OUTLIERS BY CAPPING EXTREME INPUT VALUES
# ============================================================

# Winsorization: cap extreme values at the learned IQR limits.
# Do not modify the target scores or use test data to learn limits.

X_train_corrected = X_train.clip(
    lower=lower_bounds,
    upper=upper_bounds,
    axis="columns"
)

X_test_corrected = X_test.clip(
    lower=lower_bounds,
    upper=upper_bounds,
    axis="columns"
)

print("\nOutlier correction completed using IQR capping.")
print("Training and testing data remain separate.")


# ============================================================
# 5. VISUALIZATION: OUTLIERS BEFORE CORRECTION
# ============================================================

# plt.figure(figsize=(9, 5))

# sns.barplot(
#     x=outlier_counts.index,
#     y=outlier_counts.values
# )

# plt.title("Detected Outliers Before Correction")
# plt.xlabel("Student Performance Features")
# plt.ylabel("Number of Outliers")
# plt.xticks(rotation=25, ha="right")
# plt.tight_layout()
# plt.show()


# ============================================================
# 6. VISUALIZATION: FINAL MARKS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df[TARGET],
    bins=20,
    kde=True
)

plt.title("Distribution of Final Exam Marks")
plt.xlabel("Final Exam Marks (out of 100)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 7. VISUALIZATION: CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

correlation = df[FEATURES + [TARGET]].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Student Performance Features")
plt.tight_layout()
plt.show()


# ============================================================
# 8. TRAIN THE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()
model.fit(X_train_corrected, y_train)

print("\nLinear Regression model trained successfully.")


# ============================================================
# 9. PREDICT TEST DATA
# ============================================================

y_pred = model.predict(X_test_corrected)

# Exam marks are bounded between 0 and 100.
# Keep the raw model output for calculating evaluation metrics.
y_pred_bounded = np.clip(y_pred, 0, 100)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("=" * 45)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"R-squared percentage: {r2 * 100:.2f}%")

print("\nNote: R² is not the same as prediction accuracy.")


# ============================================================
# 11. VISUALIZATION: ACTUAL VS PREDICTED MARKS
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.65
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Actual vs Predicted Final Exam Marks")
plt.tight_layout()
plt.show()


# ============================================================
# 12. FEATURE IMPORTANCE USING PERMUTATION IMPORTANCE
# ============================================================

importance_result = permutation_importance(
    model,
    X_test_corrected,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2"
)

importance = pd.Series(
    importance_result.importances_mean,
    index=FEATURES
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("-" * 45)

for feature, value in importance.items():
    print(f"{feature}: {value:.4f}")

plt.figure(figsize=(9, 5))

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance for Final Mark Prediction")
plt.xlabel("Mean Decrease in R² When Feature Is Shuffled")
plt.ylabel("Input Feature")
plt.tight_layout()
plt.show()


# ============================================================
# 13. GUI: PREDICT A NEW STUDENT'S FINAL MARKS
# ============================================================

def create_gui():

    root = tk.Tk()
    root.title("Student Performance Prediction System")
    root.geometry("760x700")
    root.minsize(680, 650)
    root.configure(bg="#F1F5F9")

    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "TLabel",
        background="#F1F5F9",
        font=("Segoe UI", 11)
    )

    style.configure(
        "TEntry",
        font=("Segoe UI", 11),
        padding=7
    )

    style.configure(
        "TButton",
        font=("Segoe UI", 11, "bold"),
        padding=9
    )

    # Header
    header = tk.Frame(root, bg="#17365D", height=100)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header,
        text="STUDENT PERFORMANCE",
        font=("Segoe UI", 20, "bold"),
        bg="#17365D",
        fg="white"
    ).pack(pady=(15, 0))

    tk.Label(
        header,
        text="Final Exam Marks Prediction",
        font=("Segoe UI", 11),
        bg="#17365D",
        fg="#DCE6F1"
    ).pack(pady=5)

    main = tk.Frame(root, bg="#F1F5F9", padx=35, pady=20)
    main.pack(fill="both", expand=True)

    tk.Label(
        main,
        text="Enter the student's details",
        font=("Segoe UI", 15, "bold"),
        bg="#F1F5F9",
        fg="#17365D"
    ).pack(anchor="w", pady=(0, 15))

    form = tk.Frame(main, bg="#F1F5F9")
    form.pack(fill="x")

    entries = {}

    for row, feature in enumerate(FEATURES):
        label = ttk.Label(form, text=feature)
        label.grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 20),
            pady=9
        )

        entry = ttk.Entry(form, width=25)
        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=9
        )

        entries[feature] = entry

    form.columnconfigure(1, weight=1)

    # Prediction output area
    result_frame = tk.Frame(
        main,
        bg="white",
        highlightbackground="#CBD5E1",
        highlightthickness=1,
        padx=15,
        pady=15
    )
    result_frame.pack(fill="x", pady=20)

    result_label = tk.Label(
        result_frame,
        text="Predicted Final Marks: -- / 100",
        font=("Segoe UI", 16, "bold"),
        bg="white",
        fg="#17365D"
    )
    result_label.pack()

    status_label = tk.Label(
        result_frame,
        text="Enter all five inputs and click Predict.",
        font=("Segoe UI", 10),
        bg="white",
        fg="#475569",
        wraplength=600
    )
    status_label.pack(pady=(8, 0))

    def predict_marks():
        try:
            values = {}

            for feature in FEATURES:
                raw_value = entries[feature].get().strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter {feature}."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"{feature} must be a finite number."
                    )

                minimum, maximum = INPUT_LIMITS[feature]

                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"{feature} must be between "
                        f"{minimum} and {maximum}."
                    )

                values[feature] = value

            new_student = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Apply the same training-derived outlier limits.
            new_student = new_student.clip(
                lower=lower_bounds,
                upper=upper_bounds,
                axis="columns"
            )

            # Use the trained model.
            raw_prediction = float(
                model.predict(new_student)[0]
            )

            predicted_marks = float(
                np.clip(raw_prediction, 0, 100)
            )

            result_label.config(
                text=f"Predicted Final Marks: "
                     f"{predicted_marks:.2f} / 100"
            )

            if predicted_marks >= 90:
                performance = "Excellent"
            elif predicted_marks >= 75:
                performance = "Very Good"
            elif predicted_marks >= 60:
                performance = "Good"
            elif predicted_marks >= 40:
                performance = "Needs Improvement"
            else:
                performance = "At Risk"

            status_label.config(
                text=f"Estimated performance category: "
                     f"{performance}\n"
                     "This is a model-based estimate, not a "
                     "guarantee of actual exam results."
            )

        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))

    buttons = tk.Frame(main, bg="#F1F5F9")
    buttons.pack(fill="x")

    ttk.Button(
        buttons,
        text="Predict Final Marks",
        command=predict_marks
    ).pack(side="left", expand=True, fill="x", padx=(0, 8))

    def clear_fields():
        for entry in entries.values():
            entry.delete(0, tk.END)

        result_label.config(
            text="Predicted Final Marks: -- / 100"
        )
        status_label.config(
            text="Enter all five inputs and click Predict."
        )

    ttk.Button(
        buttons,
        text="Clear",
        command=clear_fields
    ).pack(side="left", expand=True, fill="x", padx=(8, 0))

    footer = tk.Label(
        root,
        text="Machine Learning Project | Linear Regression",
        font=("Segoe UI", 9),
        bg="#F1F5F9",
        fg="#64748B"
    )
    footer.pack(pady=(0, 12))

    root.mainloop()


# ============================================================
# 14. START THE GUI
# ============================================================

create_gui()
