from support_vector_machine.svm import trainModel as trainSVM
from support_vector_machine.svm_smote import trainModel as trainSVMSmote
from support_vector_machine.svm_features import trainModel as trainSVMFeatures

# SVM without SMOTE
results = trainSVM('heart_failure_clinical_records_dataset.csv', 1)
print('SVM without SMOTE - Accuracy: ', results['accuracy'])
print('SVM without SMOTE - Precision: ', results['precision'])
print('SVM without SMOTE - Recall: ', results['recall'])
print('SVM without SMOTE - F-Score: ', results['fScore'])

# SVM with SMOTE
results = trainSVMSmote('heart_failure_clinical_records_dataset.csv', 1)
print('SVM with SMOTE - Accuracy: ', results['accuracy'])
print('SVM with SMOTE - Precision: ', results['precision'])
print('SVM with SMOTE - Recall: ', results['recall'])
print('SVM with SMOTE - F-Score: ', results['fScore'])

# SVM with selected important features and SMOTE
results = trainSVMFeatures('heart_failure_clinical_records_dataset.csv', 1)
print('SVM with imp features & SMOTE - Accuracy: ', results['accuracy'])
print('SVM with imp features & SMOTE - Precision: ', results['precision'])
print('SVM with imp features & SMOTE - Recall: ', results['recall'])
print('SVM with imp features & SMOTE - F-Score: ', results['fScore'])
