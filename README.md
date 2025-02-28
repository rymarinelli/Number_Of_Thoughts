# Data

The interminate dataset is hosted on Hugging Face. 
These contain the datasets used in the ablation studies. 
tiny_rl  = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_TinyR1/data/train-00000-of-00001.parquet")
deepseek = pd.read_parquet("hf://datasets/zrmarine/Chain_Of_Thought_Count_Ablation_Deepseek/data/train-00000-of-00001.parquet")
random_forest = pd.read_parquet("hf://datasets/zrmarine/DIA-Number-Of-Thoughts.csv/data/train-00000-of-00001.parquet")

