# Data

The interminate dataset is hosted on Hugging Face. 
These contain the datasets used in the ablation studies. 
 - tiny_rl  = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_TinyR1/data/train-00000-of-00001.parquet")
 - deepseek = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_Ablation_Deepseek/data/train-00000-of-00001.parquet")
 - random_forest = pd.read_parquet("hf://datasets/zrmarine/DIA-Number-Of-Thoughts.csv/data/train-00000-of-00001.parquet")

# Pickled Objects 
The trained model and a TF-IDF for processing are saved and can be loaded to 
 - joblib.dump(regressor, "thought_count_predictor.pkl")
 - joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

An easy way to use these is to load the pickled objects in and use: 

```{python}
def predict_thought_count(question, model_path="thought_count_predictor.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    """
    Predicts the thought count for a given question.

    Args:
        question (str): The input question text.
        model_path (str): Path to the saved regressor model.
        vectorizer_path (str): Path to the saved vectorizer.

    Returns:
        float: Predicted thought count.
    """

    regressor = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Transform the question using the vectorizer
    question_tfidf = vectorizer.transform([question])

    # Predict the thought count
    predicted_thought_count = regressor.predict(question_tfidf)[0]
    return predicted_thought_count
``
