class DataCleaner:
    def __init__(self):
        self.issue_texts = ["ISSUE: couldn't translate"]

    def remove_invalid_tweets(self, data):
        cleaned_data = []
        for rumor in data:
            timeline = rumor.get("timeline", [])
            # Filter the timeline
            cleaned_timeline = [entry for entry in timeline if entry[2] not in self.issue_texts]

            # Check if the cleaned timeline has valid entries
            if cleaned_timeline:
                rumor["timeline"] = cleaned_timeline
                cleaned_data.append(rumor)
        return cleaned_data