from decision_tree.dt import trainModel as trainDT
from decision_tree.dt_smote import trainModel as trainDTSmote
from decision_tree.dt_features import trainModel as trainDTFeatures

# Decision Tree without SMOTE
results = trainDT('heart_failure_clinical_records_dataset.csv', 1)
print('Decision Tree without SMOTE - Accuracy: ', results['accuracy'])
print('Decision Tree without SMOTE - Precision: ', results['precision'])
print('Decision Tree without SMOTE - Recall: ', results['recall'])
print('Decision Tree without SMOTE - F-Score: ', results['fScore'])

# Decision Tree with SMOTE
results = trainDTSmote('heart_failure_clinical_records_dataset.csv', 1)
print('Decision Tree with SMOTE - Accuracy: ', results['accuracy'])
print('Decision Tree with SMOTE - Precision: ', results['precision'])
print('Decision Tree with SMOTE - Recall: ', results['recall'])
print('Decision Tree with SMOTE - F-Score: ', results['fScore'])

# Decision Tree with selected important features and SMOTE
results = trainDTFeatures('heart_failure_clinical_records_dataset.csv', 1)
print('Decision Tree with imp features & SMOTE - Accuracy: ',
      results['accuracy'])
print('Decision Tree with imp features & SMOTE - Precision: ',
      results['precision'])
print('Decision Tree with imp features & SMOTE - Recall: ', results['recall'])
print('Decision Tree with imp features & SMOTE - F-Score: ', results['fScore'])
