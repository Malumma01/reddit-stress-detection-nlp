# reddit-stress-detection-nlp

## Project Overview
This project uses Natural Language Processing to clean and analyze Reddit posts, extract textual features with TF-IDF, and build a machine learning model to classify whether users are stressed or not. The goal is to demostrate how machine Learning can be applied to real-world mental health related text data to identify early signs of pyschological stress.

## Problem statement
Reddit users often express their thoughts and emotions through posts, but detecting stress from text can be challenging. The problem is to automatically identify whether a user is experiencing stress based on the language in their Reddit posts, enabling timely insights into mental well-being.


## Dataset

The dataset used in this project was sourced from Mental health-related Reddit post (Kaggle) and contains posts labeled to indicate whether a user is stressed or not. It consists of **2,838 posts** and **116 columns**, but only **two columns were used** for this analysis:

- `text`: The content of the Reddit post. This column was used as the feature for Natural Language Processing (NLP) because it contains the user’s expressed thoughts and emotions.
- `label`: The target variable, where 0 = Stressed and 1 = Not Stressed. This column was used for supervised classification.

## Methodology
- Text cleaning (tokenization, stopword removal, lemmatization)
- Feature extraction using TF-IDF
- Train-test split
- Model training using Logistic Regression
- Model evaluation using precision, recall, F1-score, and confusion matrix

## Results
The model achieved an accuracy of approximately 74%, with balanced performance across both classes.

## Tools & Libraries
- Python
- pandas, numpy
- nltk
- scikit-learn
- matplotlib

  ## Model Limitations & Future Improvements

While the Logistic Regression model achieved 74% accuracy, the recall for stressed posts was 70%, indicating that some stressed users were misclassified. 

Potential improvements include:
- Using class weighting or threshold adjustment to improve recall for stressed posts
- Exploring alternative models such as Bernoulli Naive Bayes, Linear SVM, or ensemble methods
- Incorporating n-grams or word embeddings to capture more contextual information
- Expanding the dataset or applying data augmentation for better generalization.

  ## Ethical Considerations

This project is for **educational and research purposes only**. The dataset used consists of publicly available Reddit posts. No personally identifiable information (PII) is included or shared.  

The model is **not intended for clinical or diagnostic use**. Predictions made by this model should **not be used as a substitute for professional mental health advice, diagnosis, or treatment**. Users experiencing stress or mental health issues are encouraged to consult a licensed professional.



## Author
Jideuno Chioma Applied Data Practitioner | Clinical intelligence & health systems

** This project demonstrates the use of Natural Language Processing and machine learning to detect and classify stress in Reddit posts, helping address real-world mental health challenges. ** 
