def collect_evidence(test_data, final_stances):

    for idx, item in enumerate(test_data):
        
        final_stance = final_stances[idx]

        if final_stance in [0, 1]:  
            evidence_entries = []
            for timeline_entry in item['top_k_timeline']:
                if timeline_entry['stance_prediction'] == final_stance:
                    evidence_entries.append(timeline_entry)
                    if len(evidence_entries) >= 5:
                        break
            item['evidence_predicted'] = evidence_entries
        else:
            # NOT ENOUGH INFO: no evidence is provided
            item['evidence_predicted'] = []
    return test_data
