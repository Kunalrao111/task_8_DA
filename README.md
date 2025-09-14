# 🏡 House Price Prediction – Feature Engineering

## 📌 Objective
Apply feature engineering techniques to improve the predictive performance of a machine learning model using the Kaggle House Prices – Advanced Regression Dataset.

## 🔧 Techniques Used
- Identified and handled missing values using `df.isnull().sum()`
- Imputed categorical nulls with `"None"` and numerical with median or zero
- Encoded ordinal features using `LabelEncoder`
- Applied `pd.get_dummies()` for one-hot encoding
- Transformed skewed numerical features using `np.log1p()`
- Evaluated feature importance using `RandomForestRegressor`
- Trained and tested model with final RMSE: **0.1506**

## 🛠️ Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## 📁 Files Included
- `task_8_DA.ipynb`: Jupyter notebook with full workflow
- `train.csv`, `test.csv`: Dataset files from Kaggle

## 📈 Result
Achieved a strong predictive model with RMSE of **0.1506** on log-transformed `SalePrice`, indicating effective feature engineering and model tuning.

---

