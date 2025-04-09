import torch
from torchvision import transforms
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

def get_model_(model_name="Salesforce/instructblip-flan-t5-xl"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    # Define image transformation (optional if you're using the processor)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, processor, image_transform, device
