import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_micro = f1_score(test_labels, predictions, average='micro')
    f1_macro = f1_score(test_labels, predictions, average='macro')
    conf_matrix = confusion_matrix(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # confusion matrix
    class_labels = ['NOT ENOUGH INFO','REFUTES','SUPPORTS']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'confusion_matrix': conf_matrix
    }
    
    



def evaluate_llm_model(true_labels, predicted_labels):
   
     
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)


    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)


    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # confusion matrix
    class_names = ['SUPPORTS','REFUTES','NOT ENOUGH INFO']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 15})
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

