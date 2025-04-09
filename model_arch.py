import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 

def get_model_(model_name= "Salesforce/instructblip-vicuna-7b"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Salesforce/instructblip-vicuna-7b"  # This is an example model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    # Define image transformation (adjust size and normalization as needed)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])

    return model, tokenizer, image_transform, device