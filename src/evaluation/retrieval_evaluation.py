import numpy as np


def get_top_k_similar_timeline_entries(similarities, rumor_id, k):
    rumor_similarities = [sim for sim in similarities if sim['rumor_id'] == rumor_id]
    sorted_similarities = sorted(rumor_similarities, key=lambda x: x['similarity'], reverse=True)
    return sorted_similarities[:k]


def evaluate_recall_at_k(test_data, similarities, k):
    
    recalls = []

    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  # Skip evaluation for "NOT ENOUGH INFO" 

        rumor_id = item['id']
        top_k_entries = get_top_k_similar_timeline_entries(similarities, rumor_id, k)

        # True relevant timeline entries (evidence)
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]

        # top K similar entries IDs
        top_k_ids = [timeline_entry['timeline_id'] for timeline_entry in top_k_entries]

        
        relevant_in_top_k = len(set(top_k_ids) & set(true_evidence_ids)) 
        total_relevant = len(true_evidence_ids)

        
        if total_relevant > 0:
            recall = relevant_in_top_k / total_relevant
            recalls.append(recall)

    # Mean recall over all evaluated rumors
    mean_recall = np.mean(recalls) if recalls else 0
    return mean_recall

def evaluate_map(test_data, similarities):
   
    average_precisions = []

    
    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  # Skip evaluation for "NOT ENOUGH INFO"

        rumor_id = item['id']
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]
        
        # Get all similarity scores for this rumor and its corresponding timeline tweets, sorted by similarity in descending order
        rumor_similarities = [sim for sim in similarities if sim['rumor_id'] == rumor_id]
        sorted_similarities = sorted(rumor_similarities, key=lambda x: x['similarity'], reverse=True)

        
        relevant_count = 0
        precision_sum = 0.0

        for i, similarity in enumerate(sorted_similarities):
            timeline_id = similarity['timeline_id']
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

