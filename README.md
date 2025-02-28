# Introduction
This repo contains Jupyternotebooks that link to Google Colab environments. The intermediate objects are hosted here with the key datasets hosted in Huggingface and in the datasets directory. 

# Notebooks 
Google Colab is used in this study to promote reproducbility.
 - Number_Of_Thought_Labeling: Annotates https://huggingface.co/datasets/TIGER-Lab/MathInstruct. The resulting annotations are used to train the random forest
 - Chain_Of_Thought_Analysis: Explores the usecases of using number of thoughts. It creates a classifer to detect malicious prompts and project injection.
 - Chain_Of_Thought_Deep_Comparison: Evalutes differences in performance across quantized Deepseek models
 - Number_Of_Thoughts_TinyR1: Evalautes TinyR1
 - Number_Of_Thoughts_Comparsion: Statistical Tests and Analysis

# Data

The interminate dataset is hosted on Hugging Face. 
These contain the datasets used in the ablation studies. 
 - tiny_rl  = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_TinyR1/data/train-00000-of-00001.parquet")
 - deepseek = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_Ablation_Deepseek/data/train-00000-of-00001.parquet")
 - random_forest = pd.read_parquet("hf://datasets/zrmarine/DIA-Number-Of-Thoughts.csv/data/train-00000-of-00001.parquet")
  

 - TinyR1 Responses are saved in TinyR1_Responses.yaml
 - Deepseek Responses are saved in Deepseek_Responses.yaml

The routing results are saved in 
 - results_baseline_4.csv
 - results_baseline_35.csv
 - results_routing_35.csv
 - results_routing_4.csv

   The results show the two combinations different thresholds tried in the number of thoughts. One attempts to segregate across three models(*_4.csv) and across 2 models(*_35.csv) The appended number is the threshold used in the study. 

Additional Files
- contains annotations used for the random forest pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count/data/train-00000-of-00001.parquet")
- inference_results.csv is in an intermeidate csv with some additional processing 

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
