def create_rumor_evidence_label_pairs(data, not_enough_info_label="NOT ENOUGH INFO"):
    pairs = []
    for item in data:
        rumor = item['rumor']
        label = item.get('label', None)
        
        # Use evidence or fallback to timeline if no evidence is available
        if len(item['evidence']) > 0:
            for evidence_entry in item['evidence']:
                evidence_text = evidence_entry[2]
                pairs.append({'rumor': rumor, 'evidence': evidence_text, 'label': label})
        else:
            # If no evidence is available, use up to 4 timeline entries
            for i, timeline_entry in enumerate(item['timeline'][:4]):
                timeline_text = timeline_entry[2]
                pairs.append({'rumor': rumor, 'evidence': timeline_text, 'label': not_enough_info_label})
    
    return pairs
 
def prepare_dataset_for_classification(data):
    rumors, evidences, labels = [], [], []
    for item in data:
        rumors.append(item['rumor'])
        evidences.append(item['evidence'])
        labels.append(item['label'])
    combined_text = [f"{rumor} {evidence}" for rumor, evidence in zip(rumors, evidences)]
    return combined_text, labels


def prepare_dataset_for_classification_llm_roberta(data):
    rumors, evidences, labels = [], [], []
    for item in data:
        rumors.append(item['rumor'])
        evidences.append(item['evidence'])
        labels.append(item['label'])
    combined_text = [f"Rumor: {rumor} </s> Evidence: {evidence}" for rumor, evidence in zip(rumors, evidences)]
    
    return combined_text, labels


def prepare_dataset_for_classification_llm(data):
    rumors, evidences, labels = [], [], []
    for item in data:
        rumors.append(item['rumor'])
        evidences.append(item['evidence'])
        labels.append(item['label'])
    combined_text = [f"Rumor: {rumor} [SEP] Evidence: {evidence}" for rumor, evidence in zip(rumors, evidences)]

    return combined_text, labels