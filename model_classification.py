import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import re
from functools import reduce
import mlflow
import mlflow.sklearn  # To track the scikit-learn model
import pickle

# Dictionary of treated languages
selected_lang = {
    "ar": "Arabic / العربية", "as": "Assamese / অসমীয়া", "be": "Belarusian / Беларуская",
    "bo": "Tibetan / བོད་ཡིག", "dv": "Maldivian / ދިވެހި", "fr": "French", "fa": "Persian / فارسی",
    "hi": "Hindi / हिन्दी", "ja": "Japanese / 日本語", "ka": "Georgian / ქართული",
    "ko": "Korean / 한국어", "ru": "Russian / русский", "te": "Telugu / తెలుగు",
    "zh": "Chinese / 中文"
}

similar_lang = ["ru", "be", "ca", "es", "bs", "sh", "ce"]
path = "../../CORPORA_pp/"


# Function to preprocess text data, returning a list of texts and their associated language codes
def preprocessing(dico):
    corpus_text = []
    corpus_languages = []

    # Process each language corpus
    for code_lang in dico.keys():
        with open(path + "pp_" + code_lang.upper() + ".txt", "r", encoding="utf-8") as lang_corpus_file:
            lang_corpus_text = lang_corpus_file.read()
            preprocessed_clean_text = lang_corpus_text.split("\n")

            # Clean and append sentences, removing unwanted characters
            for sent in preprocessed_clean_text[:2000]:
                corpus_text.append(re.sub(r'[\[\]\n\t0-9]+', '', sent.strip()))

            # Associate each sentence with the language code
            corpus_languages.append([code_lang] * len(preprocessed_clean_text[:2000]))

    # Flatten the list of languages
    corpus_languages = reduce(lambda x, y: x + y, corpus_languages)

    return corpus_text, corpus_languages


# Function to classify languages, using Naive Bayes, and track the experiment with MLflow
def model_classification(X, y, dico):
    # Start an MLflow run for tracking the experiment
    with mlflow.start_run() as run:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose vectorizer type based on whether languages are in the similar_lang list
        vectorizer = None
        if any(lang in similar_lang for lang in dico.keys()):
            # Use word-level vectorization for similar languages
            vectorizer = CountVectorizer(analyzer='word', decode_error='ignore')
            mlflow.log_param("vectorizer_type", "word")
        else:
            # Use character-level vectorization for other languages
            vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), decode_error='ignore')
            mlflow.log_param("vectorizer_type", "char_wb")
            mlflow.log_param("ngram_range", (2, 2))

        # Build the pipeline with CountVectorizer and Multinomial Naive Bayes
        pipeline = Pipeline([('vectorizer', vectorizer), ('model', MultinomialNB())])

        # Train the model
        model = pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_predicted = pipeline.predict(X_test)

        # Log accuracy as a metric in MLflow
        accuracy = np.mean(y_predicted == y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model in MLflow
        mlflow.sklearn.log_model(model, "model")

        # Generate confusion matrix and classification report
        confusion_mat = confusion_matrix(y_test, y_predicted)
        print(f"Confusion matrix:\n{confusion_mat}\n")
        report = classification_report(y_test, y_predicted)
        print(report)

        # Log the classification report to MLflow
        mlflow.log_text(report, "classification_report_1.txt")

        # Optionally save the model locally with pickle (if needed for further processing)
        with open("model_nb_1.pkl", "wb") as file_pickle:
            pickle.dump(model, file_pickle)

    return report


# Main execution to preprocess data, train the model, and track results with MLflow
if __name__ == "__main__":
    # Preprocess data from the language corpora
    X, y = preprocessing(selected_lang)

    # Train the model and track the results in MLflow
    model_classification(X, y, selected_lang)
