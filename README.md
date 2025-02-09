# Rumor Verification Using Evidence from Authorities

This project addresses the challenge of rumor verification on social media, specifically on Twitter (now known as Platform X), by leveraging evidence from authoritative accounts in both Arabic and English. Our proposed system is designed as a three-stage process that includes **evidence retrieval**, **stance detection**, and **rumor verification**. The system architecture is built upon advanced transformer-based models.
## Project Overview

1. **Evidence Retrieval**: We utilize the SBERT (Sentence-BERT) model to generate dense embeddings for rumors and their corresponding timeline tweets. This process captures semantic meanings in sentences, enabling efficient comparison through cosine similarity. The top-ranking tweets are selected as evidence candidates based on their relevance to the rumor.

2. **Stance Detection**: To classify the stance of each evidence candidate with respect to the rumor, we fine-tuned the multilingual XLM-RoBERTa model on a bilingual dataset of English and Arabic samples. This model categorizes evidence as either:
   - **SUPPORTS**
   - **REFUTES**
   - **NOT ENOUGH INFO**

3. **Rumor Verification**: We determine the final stance of the rumor by aggregating the stance predictions of each evidence candidate using weighted voting. This approach combines the individual stances to produce a final rumor label.

## Methodology and Comparisons

To evaluate the effectiveness of our approach, we compared the results of each stage with traditional machine learning methods:

- **Evidence Retrieval**: SBERT was compared with bag-of-words techniques, including TF-IDF and BM25. SBERT excelled in English evidence retrieval, while BM25 was more effective for Arabic.
  
- **Stance Detection**: The fine-tuned XLM-RoBERTa model was tested against a mix of traditional machine learning models and other large language models, showing superior performance in both languages.

- **Rumor Verification**: Weighted voting was used as the primary aggregation method and was compared with majority and soft voting methods. Weighted voting achieved the highest accuracy across English and Arabic datasets.

## Model Availability

The fine-tuned models, including **XLM-RoBERTa** and **KEMLM**, are available for download in a compressed ZIP file on Google Drive. You can download them using the link below:

- **[Download Fine-tuned Models (XLM-RoBERTa and KEMLM)](https://drive.google.com/file/d/1PdqVoKVTvKeZJJ1ND9IDEdgh60Ej0ImN/view?usp=share_link)**

After downloading, unzip the file and place the extracted models in the `src/traind_models/` directory within the project structure.

## Setup and Dependencies

1. **Set Up the Virtual Environment and Install Requirements**:
   Ensure that the required dependencies are installed by running:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # for macOS/Linux
   venv\Scripts\activate     # for Windows
   pip install -r requirements.txt

2. **Adjust Data Paths**:
Before running the notebooks, ensure the data paths are correctly set to match the project structure. Open each notebook and update the data paths as needed to match your local setup (e.g., `data/` directory for training datasets or `traind_models/` directory for saved models). This step is necessary to load data and model files correctly.

3. **Run the Notebooks**:
After adjusting the data paths, you can run the notebooks to explore each stage of the model, including evidence retrieval, stance detection, and rumor verification.