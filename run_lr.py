from logistic_regression.lr import trainModel as trainLR
from logistic_regression.lr_smote import trainModel as trainLRSmote
from logistic_regression.lr_features import trainModel as trainLRFeatures

# Logistic regression without SMOTE
# results = trainLR('heart_failure_clinical_records_dataset.csv', 1)
# print('Logistic Regression - Accuracy: ', results['accuracy'])
# print('Logistic Regression - Precision: ', results['precision'])
# print('Logistic Regression - Recall: ', results['recall'])
# print('Logistic Regression - F-Score: ', results['fScore'])

# Logistic regression with SMOTE
# results = trainLRSmote('heart_failure_clinical_records_dataset.csv', 1)
# print('Logistic Regression with SMOTE - Accuracy: ', results['accuracy'])
# print('Logistic Regression with SMOTE - Precision: ', results['precision'])
# print('Logistic Regression with SMOTE - Recall: ', results['recall'])
# print('Logistic Regression with SMOTE - F-Score: ', results['fScore'])

# Logistic regression with selected important features and SMOTE
results = trainLRFeatures('heart_failure_clinical_records_dataset.csv', 1)
print('Logistic Regression with imp features & SMOTE - Accuracy: ',
      results['accuracy'])
print('Logistic Regression with imp features & SMOTE - Precision: ',
      results['precision'])
print('Logistic Regression with imp features & SMOTE - Recall: ',
      results['recall'])
print('Logistic Regression with imp features & SMOTE - F-Score: ',
      results['fScore'])
