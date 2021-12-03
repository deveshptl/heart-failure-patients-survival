from random_forest.rf import trainModel as trainRF
from random_forest.rf_smote import trainModel as trainRFSmote
from random_forest.rf_features import trainModel as trainRFFeatures


# Random Forest without SMOTE
results = trainRF('heart_failure_clinical_records_dataset.csv', 1)
print('Random Forest without SMOTE - Accuracy: ', results['accuracy'])
print('Random Forest without SMOTE - Precision: ', results['precision'])
print('Random Forest without SMOTE - Recall: ', results['recall'])
print('Random Forest without SMOTE - F-Score: ', results['fScore'])

# Random Forest with SMOTE
results = trainRFSmote('heart_failure_clinical_records_dataset.csv', 1)
print('Random Forest with SMOTE - Accuracy: ', results['accuracy'])
print('Random Forest with SMOTE - Precision: ', results['precision'])
print('Random Forest with SMOTE - Recall: ', results['recall'])
print('Random Forest with SMOTE - F-Score: ', results['fScore'])

# Random Forest with selected important features and SMOTE
results = trainRFFeatures('heart_failure_clinical_records_dataset.csv', 1)
print('Random Forest with imp features & SMOTE - Accuracy: ',
      results['accuracy'])
print('Random Forest with imp features & SMOTE - Precision: ',
      results['precision'])
print('Random Forest with imp features & SMOTE - Recall: ', results['recall'])
print('Random Forest with imp features & SMOTE - F-Score: ', results['fScore'])
