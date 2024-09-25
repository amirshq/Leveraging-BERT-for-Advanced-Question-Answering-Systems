# Import the models and save model paths 
from transformers import AutoTokenizer, AutoModel

# Define your model paths
model_paths = {
    'bert-base-uncased': '/Users/amirshahcheraghian/BERT Models/bert-base-uncased/',
    'bert-large-uncased-whole-word-masking-finetuned-squad': '/Users/amirshahcheraghian/BERT Models/bert-large-uncased-whole-word-masking-finetuned-squad/',
    'albert-large-v2': '/Users/amirshahcheraghian/BERT Models/albert-large-v2/',
    'albert-base-v2': '/Users/amirshahcheraghian/BERT Models/albert-base-v2/'
}
#'distilbert': '/Users/amirshahcheraghian/BERT Models/distilbert/'
# Function to download and save model
def download_and_save_model(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

# Download and save each model
for model_name, path in model_paths.items():
    download_and_save_model(model_name, path)
    print(f"Model {model_name} saved to {path}")
