# Water Quality Prediction

A machine learning project that predicts whether a water sample is **safe to drink** based on 20 chemical and biological measurements. Six classification models are trained on the data and compared; Random Forest gives the best accuracy (about 96.9%).

The full work is in one notebook: [`water_quality_predicting.ipynb`](water_quality_predicting.ipynb).

---

## Dataset

- **File:** `waterQualityPrediction.csv`
- **Size:** 7,999 rows and 21 columns (20 features and 1 target)
- **Target:** `is_safe` (`1` = safe, `0` = not safe), renamed to `potability` in the notebook
- **Features:** aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, flouride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, uranium

**Data issues found:**
- Three rows contain `#NUM!` in both the `ammonia` and `is_safe` columns. This made both columns load as text, so the rows were removed and the columns converted to numbers. **7,996 samples** remain.
- No other missing values.
- The classes are imbalanced: 7,084 not safe vs. 912 safe (about 11% safe).

## Steps in the Notebook

1. **Load and inspect** the data (shape, head/tail, info, summary statistics).
2. **Clean** the data: remove the `#NUM!` rows, fix data types, rename `is_safe` to `potability`, check for null values (none).
3. **Handle outliers:** box plots show outliers in several features. Values in `aluminium`, `arsenic` and `nitrites` are capped at the 1.5 × IQR upper and lower limits.
4. **Visualise the data:** correlation heatmap, class-distribution pie chart, pair plots, histograms and scatter plots (coloured by potability).
5. **Balance the classes** with SMOTE, giving 7,084 samples in each class (14,168 in total).
6. **Split** the data 80% train / 20% test (`random_state=42`): 11,334 training and 2,834 test samples.
7. **Train and evaluate** six models using accuracy, confusion matrix and classification report:
   - Logistic Regression
   - Decision Tree (`max_depth` tested from 1 to 10)
   - Random Forest
   - K-Nearest Neighbours (`n_neighbors` tested from 4 to 10, k = 4 chosen)
   - Support Vector Machine (RBF kernel)
   - Gaussian Naive Bayes
8. **Compare** the models in a table and bar chart.

## Results

Accuracy on the test set:

| Model | Accuracy |
|---|---|
| **Random Forest** | **96.86%** |
| Decision Tree | 96.01% |
| K-Nearest Neighbours (k = 4) | 87.30% |
| Logistic Regression | 80.45% |
| Gaussian Naive Bayes | 80.42% |
| Support Vector Machine (RBF) | 76.85% |

Random Forest also had the strongest class-wise scores, with precision and recall between 0.96 and 0.98 for both classes.

## Limitations

- SMOTE is applied to the whole dataset **before** the train/test split, so synthetic samples may leak into the test set and the scores may be higher than real-world performance. Splitting first and applying SMOTE only to the training set would avoid this.
- Features are not scaled (`StandardScaler` is imported but not used), which affects KNN, SVM and Logistic Regression.
- The Decision Tree depth and the KNN `k` were chosen by comparing test-set accuracy rather than using cross-validation.
- Gaussian Naive Bayes is evaluated in the notebook but is not included in the final comparison table and chart.

## Author

**Your Name** · [GitHub](https://github.com/your-username)
