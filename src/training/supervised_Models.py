from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


#svm model 
def build_svm_model(train_features, train_labels):
    
    svm_model = SVC(kernel='linear',class_weight='balanced')
    svm_model.fit(train_features, train_labels)
    return svm_model

# random forest model
def build_random_forest_model(train_features, train_labels):
   
    random_forest_model = RandomForestClassifier( class_weight='balanced', random_state=42)
    random_forest_model.fit(train_features, train_labels)
    return random_forest_model