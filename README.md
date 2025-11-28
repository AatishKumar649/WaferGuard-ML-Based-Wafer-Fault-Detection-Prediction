# WaferGuard-ML-Based-Wafer-Fault-Detection-Prediction

Problem statement 

During semiconductor manufacturing, wafers are instrumented with many sensors to monitor process variables. Manual inspection of suspected faulty wafers is time-consuming and can cause production stoppages and costs. The problem addressed: given the sensor measurements for a wafer, classify whether the wafer is faulty (requires further inspection/rework) or non-faulty, with a special emphasis on minimizing false negatives (i.e., wafers that are faulty but predicted as good).
Why this matters: reducing false negatives saves production time and energy, reduces material waste, and lowers inspection manpower — improving throughput and lowering cost per wafer.


Dataset & setup 

Dataset loaded in notebook: CSV from the project repository: wafer_23012020_041211.csv (loaded directly from GitHub URL in the notebook).
Rows: each row corresponds to a single wafer (sample).
Columns: numerous sensor readings (numerical); one label column indicates wafer status (Good/Bad or numeric equivalent).
Main challenges in the data: high dimensionality, redundant and constant columns, possible missing values, class imbalance, and noisy sensors.

Design and methodology — step-by-step 
       Data loading & initial inspection
Read CSV into a pandas DataFrame (pd.read_csv(...)).
Viewed head(), tail(), info(), describe() to understand data types, ranges, and null counts.
Confirmed the target column and value distribution with value_counts().
Why: baseline understanding to drive cleaning decisions and detect obvious issues (nulls, duplicates, constants).

       Column pruning (feature reduction steps you implemented)
You created helper functions to automatically drop non-informative columns:
get_cols_zero_std(df) — find columns with zero standard deviation (constants) and remove them.
Rationale: constant columns carry no predictive information and can increase processing time.
get_reduntant_col(df, missing_thresh=0.7) — identify and drop columns with a very high percentage of missing values (threshold used in notebook: 70%).
Rationale: columns with mostly missing values can be noisy and should be removed rather than imputed blindly.
Combined final cols_to_drop = cols_drop_1 + cols_drop_2 + ['Wafers'] and dropped them from the DataFrame.
Why: these steps reduce dimensionality and improve model robustness.

       Train/test split
Used sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42) to hold out 20% as test set.
This ensures a clean final evaluation set not used during preprocessing tuning or cross-validation.
Why: avoid data leakage and provide an honest test evaluation.

      Exploratory Data Analysis (EDA) & clustering
Visualized feature distributions (histograms/KDEs), boxplots for outliers, correlation heatmap to inspect multicollinearity.
Performed clustering/segmentation (the notebook shows a wafer_clus array and cluster counts for different cluster labels).
Purpose: to discover natural groups in the sensor data which can explain different failure modes or latent behavior and to inform modeling strategy.
Why: EDA reveals structure, outliers, and relationships that impact feature engineering and model choice.

      Missing value handling & imputation
Checked missing counts with df.isnull().sum().
Imputation strategy (implemented in notebook pipeline): choose suitable imputation (median or other) for numeric sensor columns rather than mean when outliers present.
Why: imputation preserves rows and avoids dropping valuable samples while handling corrupted sensors.

      Resampling / Class imbalance
The notebook demonstrates resampling steps and prints shape before & after resampling:
print("Before resampling, Shape of training instances: ", np.c_[X, y].shape)
print("After resampling, Shape of training instances: ", np.c_[X_res, y_res].shape)
Used resampling (SMOTE or similar synthetic oversampling technique) to balance classes and printed the resulting target category counts (np.unique(y_res)).
Why: wafer faults are typically rarer than non-faulty wafers; balancing is critical to improve recall on the faulty class and reduce false negatives.

       Preprocessing pipeline
Built reusable preprocessing using sklearn.pipeline.Pipeline (pipeline imported and used in notebook).
Typical pipeline stages you used:
Imputer (median or custom)
Scaler (StandardScaler or MinMax)
Optional dimensionality reduction (PCA) or feature selector
The pipeline was fit on training data and applied to test data (X_train_prep, X_test_prep in notebook).
Why: pipelines ensure reproducibility and prevent leakage of test statistics into training.

       Model selection & training (what you ran)
You trained and compared the following classifiers (explicit imports appear in the notebook):
Support Vector Classifier (SVC) — sklearn.svm.SVC
Why used: strong baseline for classification with regularization; useful on medium dimensional data.
Random Forest Classifier — sklearn.ensemble.RandomForestClassifier
Why used: robust, interpretable via feature importances, handles noisy/heterogeneous features well.
XGBoost Classifier — xgboost.XGBClassifier
Why used: powerful gradient boosting classifier, often top-performing for tabular data.
Process used:
Fit each model using training data (after resampling/preprocessing).
Used cross_val_predict and cross_val_score for cross-validated predictions and metric stability.
For XGBoost, obtained cross-validated predictions cross_val_predict(xgb_clf, X_test_prep, y_test_prep_xgb, cv=5) and computed AUC score using roc_auc_score.
Why: comparing multiple model families and using cross-validation helps find the best generalizable classifier and reduces overfitting.

       Evaluation metrics & reporting
Primary evaluation metric used in notebook: ROC-AUC (roc_auc_score) — reported for XGB with cross-validated predictions.
Additional metrics recommended: Precision, Recall (especially for the faulty class), F1-score, and Confusion Matrix (to detail false negatives vs false positives). Use sklearn.metrics.classification_report and confusion_matrix to gather these.
Important: in manufacturing settings, Recall for the faulty class (minimize false negatives) is often more critical than overall accuracy.

