# Improving the Prediction of Heart Failure Patients’ Survival Using SMOTE and Effective Data Mining Techniques

This repository is the implementation of [https://ieeexplore.ieee.org/document/9370099](https://ieeexplore.ieee.org/document/9370099)

### Instructions

1. To run this project, clone or download this repository.
2. Open terminal and navigate to the folder where this repository is downloaded.
3. Run `pip install -r requirements.txt` to install the required libraries.
4. Now, execute either of the following commands to run specific classification model.

   For **Decision Tree**, run `python run_dt.py`

   For **Extra Tree Classifier**, run `python run_etc.py`

   For **Logistic Regression**, run `python run_lr.py`

   For **Random Forest**, run `python run_rf.py`

   For **SVM**, run `python run_svm.py`

Each of the above commands will train the model by:

1. not handling class imbalance.
2. handling class imbalance using SMOTE
3. selecting important features and using SMOTE for class imbalance.

_Note_: `notebooks/` folder contains temporary work done in jupyter notebooks for exploration purposes, and `analysis results.txt` file contains the best results obtained so far on each model.

<h3 align='center'>
HAVE FUN
</h3>
