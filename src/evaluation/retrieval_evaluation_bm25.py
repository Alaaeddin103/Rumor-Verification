import numpy as np


def get_top_k_bm25_timeline_entries(test_data, rumor_id, k):
    rumor_data = [item for item in test_data if item['id'] == rumor_id]
    if not rumor_data:
        return []

    sorted_timeline_entries = sorted(rumor_data[0]['timeline'], key=lambda x: x[-1], reverse=True)  
    return sorted_timeline_entries[:k]


def evaluate_recall_at_k_bm25(test_data, k):
    
    recalls = []
    
    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  # Skip evaluation for "NOT ENOUGH INFO"

        rumor_id = item['id']
        top_k_entries = get_top_k_bm25_timeline_entries(test_data, rumor_id, k)

        # True relevant timeline entries (evidence)
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]

        # Extract IDs from top K 
        top_k_ids = [timeline_entry[1] for timeline_entry in top_k_entries]

        # Calculate recall
        relevant_in_top_k = len(set(top_k_ids) & set(true_evidence_ids))  
        total_relevant = len(true_evidence_ids)

        if total_relevant > 0:
            recall = relevant_in_top_k / total_relevant
            recalls.append(recall)

    mean_recall = np.mean(recalls) if recalls else 0
    return mean_recall


def evaluate_map_bm25(test_data):
    
    average_precisions = []

    for item in test_data:
        if item['label'] == "NOT ENOUGH INFO":
            continue  # Skip evaluation for "NOT ENOUGH INFO"

        rumor_id = item['id']
        true_evidence_ids = [evidence_entry[1] for evidence_entry in item['evidence']]
        
        # Get BM25 scores for this rumor's timeline entries, sorted by BM25 score in descending order
        top_k_entries = get_top_k_bm25_timeline_entries(test_data, rumor_id, len(item['timeline']))

        relevant_count = 0
        precision_sum = 0.0

        for i, timeline_entry in enumerate(top_k_entries):
            timeline_id = timeline_entry[1]  
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