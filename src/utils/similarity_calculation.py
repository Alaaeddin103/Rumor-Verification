import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarities(data):
    similarities = []
    for item in data:
        rumor_vector = np.array(item['rumor_vector'])

        # Calculate similarities with timeline entries
        for timeline_entry in item['timeline']:
            timeline_vector = np.array(timeline_entry[-1])  
            similarity = cosine_similarity(rumor_vector.reshape(1, -1), timeline_vector.reshape(1, -1))[0][0]

            similarities.append({
                'rumor_id': item['id'],
                'rumor_text': item['rumor'],
                'timeline_id': timeline_entry[1],
                'timeline_text': timeline_entry[2],
                'similarity': similarity
            })
    return similarities


