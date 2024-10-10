
# Language Classification with Naive Bayes and MLflow

This project classifies written text into 14 languages using a **Naive Bayes** classifier with the help of **scikit-learn** and **MLflow** for tracking experiments. The model is trained on preprocessed text data and uses either word-level or character-level n-grams for vectorization, depending on the language being processed.

## Features
- **MLflow Integration**: Automatically tracks parameters, metrics, and model artifacts.
- **Preprocessing**: Cleans and processes text data for multiple languages.
- **Naive Bayes Classification**: Uses the Multinomial Naive Bayes algorithm for text classification.
- **Parameter Logging**: Logs important parameters like the vectorizer type and n-gram range.
- **Metrics Logging**: Tracks and logs the accuracy and classification report.

## Project Structure

```bash
.
├── model_classification.py     # Main Python script for training and tracking
├── requirements.txt            # Dependencies required to run the project
├── model_nb.pkl                # Optional: Saved model (if needed locally)
└── CORPORA_pp/                 # Directory with language text files (not included in repo)
```

## Setup

1. Clone the repository and navigate into the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your preprocessed language text files in the `CORPORA_pp/` directory (not included in the repository).

## Running the Code

1. Run the Python script to preprocess the data, train the model, and track results in MLflow:
   ```bash
   python model_classification.py
   ```

2. After running the script, launch the **MLflow UI** to view experiment details:
   ```bash
   mlflow ui
   ```

   Open `http://localhost:5000` in your browser to access the MLflow dashboard, where you can see parameters, metrics, and model artifacts.

## Customization

- You can modify the `selected_lang` dictionary to include additional languages and their corresponding text data.
- You can change the vectorization type based on the similarity of the languages (using word-based or character-based n-grams).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
