from extra_tree_classifier.etc import trainModel as trainETC
from extra_tree_classifier.etc_smote import trainModel as trainETCSmote
from extra_tree_classifier.etc_features import trainModel as trainETCFeatures

# Extra Tree Classifier without SMOTE
results = trainETC('heart_failure_clinical_records_dataset.csv', 1)
print('Extra Tree Classifier - Accuracy: ', results['accuracy'])
print('Extra Tree Classifier - Precision: ', results['precision'])
print('Extra Tree Classifier - Recall: ', results['recall'])
print('Extra Tree Classifier - F-Score: ', results['fScore'])

# Extra Tree Classifier with SMOTE
results = trainETCSmote('heart_failure_clinical_records_dataset.csv', 1)
print('Extra Tree Classifier with SMOTE - Accuracy: ', results['accuracy'])
print('Extra Tree Classifier with SMOTE - Precision: ', results['precision'])
print('Extra Tree Classifier with SMOTE - Recall: ', results['recall'])
print('Extra Tree Classifier with SMOTE - F-Score: ', results['fScore'])

# Extra Tree Classifier with SMOTE and selected important features
results = trainETCFeatures('heart_failure_clinical_records_dataset.csv', 1)
print('Extra Tree Classifier with imp features & SMOTE - Accuracy: ', results['accuracy'])
print('Extra Tree Classifier with imp features & SMOTE - Precision: ', results['precision'])
print('Extra Tree Classifier with imp features & SMOTE - Recall: ', results['recall'])
print('Extra Tree Classifier with imp features & SMOTE - F-Score: ', results['fScore'])