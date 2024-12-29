# MIT-BIH Arrhythmia Classification with Machine Learning

## 🚀 Overview
This project explores the classification of arrhythmias using advanced machine learning algorithms, leveraging the MIT-BIH Arrhythmia Database. By applying cutting-edge techniques and rigorous evaluation metrics, the study aims to enhance automated cardiac rhythm classification and contribute to advancements in medical diagnostics.

## 🧑‍💻 Key Features
- **Algorithms:** Implementation and evaluation of Random Forest, Gradient Boosting, and XGBoost classifiers.
- **Performance:** Detailed performance analysis with and without optimization using metrics like accuracy, precision, recall, F1 Score, and AUC.
- **Data-Driven Insights:** Comprehensive signal processing and feature extraction from ECG data.
- **Optimization:** Hyperparameter tuning for enhanced model performance using GridSearchCV.

## 📊 Results
- **Random Forest:** Achieved top accuracy of 85.76% (pre-optimization) and 85.16% (optimized).
- **XGBoost:** Demonstrated robust accuracy of 85.07% (pre-optimization) and 85.84% (optimized).
- **Gradient Boosting:** Achieved a modest accuracy of 75.63% (pre-optimization) and 80.69% (optimized).

## 📋 Methodology
1. **Dataset:** Used the MIT-BIH Arrhythmia Database comprising 48 annotated ECG recordings.
2. **Signal Processing:**
   - Noise reduction with bandpass filtering (0.5 Hz to 40 Hz).
   - Sliding window segmentation for capturing rhythm variations.
   - Z-score normalization to standardize data.
3. **Feature Extraction:**
   - Time-domain: Statistical metrics, heart rate, amplitude features.
   - Frequency-domain: Power spectral density using Welch’s method.
4. **Model Selection:** Compared performance of Random Forest, Gradient Boosting, and XGBoost classifiers.
5. **Optimization:** Hyperparameter tuning to enhance accuracy and reduce overfitting.

## 📈 Evaluation Metrics
- **Accuracy:** Overall prediction correctness.
- **Precision:** Reliability of positive predictions.
- **Recall:** Ability to identify true positives.
- **F1 Score:** Balance between precision and recall.
- **AUC:** Model's ability to discriminate between classes.

## 🛠 Tools & Libraries
- **Data Processing:** `WFDB`, `NumPy`, `Pandas`, `SciPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
### ⭐️ **If you find this repository useful, give it a star!**
