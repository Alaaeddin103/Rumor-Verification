import numpy as np


def evaluate_recall_evidence(test_data):
    
    recalls = []

    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  


        # True relevant timeline entries (evidence)
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]

        # predicted evidence ids
        evidence_predicted_ids = [entry['timeline_id'] for entry in item['evidence_predicted']]

        
        relevant_predicted_evidence = len(set(evidence_predicted_ids) & set(true_evidence_ids)) 
        total_relevant = len(true_evidence_ids)

        
        if total_relevant > 0:
            recall = relevant_predicted_evidence / total_relevant
            recalls.append(recall)

    # Mean recall over all evaluated rumors
    mean_recall = np.mean(recalls) if recalls else 0
    return mean_recall

def evaluate_map_evidence(test_data):
    
    average_precisions = []

    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  # Skip evaluation for "NOT ENOUGH INFO" 

        # Ground truth evidence IDs (e.g., tweet IDs)
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]

        # Predicted evidence ids
        predicted_evidence_ids = [entry['timeline_id'] for entry in item['evidence_predicted']]

        relevant_count = 0  
        precision_sum = 0.0  

        
        for i, timeline_id in enumerate(predicted_evidence_ids):
            if timeline_id in true_evidence_ids:
                relevant_count += 1
                precision_at_rank = relevant_count / (i + 1)  
                precision_sum += precision_at_rank

        if relevant_count > 0:
            
            average_precision = precision_sum / len(true_evidence_ids)
        else:
            average_precision = 0.0

        average_precisions.append(average_precision)

    
    mean_average_precision = np.mean(average_precisions) if average_precisions else 0
    return mean_average_precision