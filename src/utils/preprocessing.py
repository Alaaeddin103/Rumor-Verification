
def preprocess_data(data, preprocessor):
    preprocessed_data = []
    for item in data:
        preprocessed_item = item.copy()
        preprocessed_item['rumor'] = preprocessor.preprocess_text(item['rumor'])

        # Preprocess each timeline and evidence entry
        for i, timeline_entry in enumerate(item['timeline']):
            preprocessed_item['timeline'][i][2] = preprocessor.preprocess_text(timeline_entry[2])

        for i, evidence_entry in enumerate(item['evidence']):
            preprocessed_item['evidence'][i][2] = preprocessor.preprocess_text(evidence_entry[2])

        preprocessed_data.append(preprocessed_item)
    return preprocessed_data