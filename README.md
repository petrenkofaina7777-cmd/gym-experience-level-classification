# Predicting a gym member's experience level (KNN vs. Decision Tree)

Multi-class classification of `Experience_Level` (1 = beginner, 2 = intermediate, 3 = advanced) from training and physiological parameters, built with scikit-learn.

**Data:** [Gym Members Exercise Dataset on Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) (973 rows, 15 columns, no missing values).
Download `gym_members_exercise_tracking.csv` and put it next to the notebook.

## What is in the notebook
- Exploratory data analysis: class balance, descriptive statistics, correlations, skewness, histograms / KDE / box-plots
- Leak-free preprocessing: `Pipeline` + `ColumnTransformer` (standard scaling, one-hot encoding), stratified 75/25 split
- Two models: K-Nearest Neighbors and Decision Tree, evaluated with 10-fold stratified cross-validation
- Hyper-parameter tuning with `GridSearchCV` **on the training set only**, one final evaluation on the test set
- Interpretation: tree visualisation, decision rules, permutation feature importance

## Results (test set, 244 samples)

| Model | CV accuracy | Test accuracy |
|---|---|---|
| KNN (k=7), baseline | 0.833 ± 0.050 | 0.852 |
| Decision Tree, unrestricted (depth 17) | 0.851 ± 0.045 | 0.828 |
| KNN (k=21, distance-weighted), tuned | 0.859 ± 0.036 | 0.865 |
| **Decision Tree (max_depth=4), tuned** | **0.896 ± 0.036** | **0.922** |

## Key findings
- Restricting the tree depth (not switching algorithms) gave the biggest improvement: 0.828 → 0.922.
- Only `Session_Duration` and `Workout_Frequency` carry signal; age, heart rate, weight, etc. have no effect (permutation importance ≈ 0).
- Class 3 is classified perfectly; all remaining errors of the tree are class 1 predicted as class 2 (people training 3×/week appear in both classes).
- The dataset looks synthetic (the label is almost a function of two features), so the high accuracy should not be generalised to real gym data.

## Run it
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook gym_experience_level_classification.ipynb
```

The full written report (in Slovak) accompanies this project.
