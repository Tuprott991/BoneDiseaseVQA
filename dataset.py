import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import csv
import random
from transformers import AutoTokenizer
import os

BASE_IMAGE_PATH = "/kaggle/input/bonevqa/DemoBoneData"


class BoneVQADataset(Dataset):
    def __init__(self, csv_file, data_file, tokenizer, image_transform, max_length=512, num_questions=10):
        """
        Args:
            data_file (str): Path to a JSONL file containing the dataset.
            tokenizer: Tokenizer for text input.
            image_transform: Transformation pipeline for images.
            max_length (int): Maximum token length for the text inputs.
        """
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)  # Use load() instead of loads()

        # If samples is not already a list, but something like {"data": [...]}
        # you might need to extract the actual data array
        if isinstance(self.samples, dict) and "data" in self.samples:
            self.samples = self.samples["data"]

        self.questions = []
        # Read All questions from the CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row['\ufeffQuestion'])

        # Print all questions
        # print("All questions:")
        # for i, question in enumerate(self.questions):
        #     print(f"Q{i+1}: {question}")
        
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.num_questions = num_questions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load sample
        sample = self.samples[idx]
        image_path = os.path.join(BASE_IMAGE_PATH, sample["image_url"])
        # clinical_text = sample["clinical_text"]

        selected_questions = random.sample(self.questions, self.num_questions)

        if self.num_questions == 1:
            input_questions  = f"{selected_questions[0]}"
        else:
            input_questions  = " ".join([f"Q{i+1}: {q}" for i, q in enumerate(selected_questions)])

        diagnose = sample["diagnose"]
        condition = sample["condition"]

        # Anwser for the model
        input_text = f"Câu hỏi: {input_questions}"
        # print(input_text)
        # # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)     

        # Output
        answer = f"Chẩn đoán: {diagnose} [SEP] Tình trạng: {condition}"

        # Concatenate clinical context and question into a single input text.
        # You can use special tokens or delimiters to separate them.

        # Print a sample for debugging
        # print(f"Sample {idx}:")
        # print(f"Image path: {image_path}")
        # print(f"Input text: {input_text}")
        # print(f"Answer: {answer}")
        # print(f"Condition: {condition}")
        

        # Tokenize input text and target answer
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        targets = {k: v.squeeze(0) for k, v in targets.items()}

        return {
            "pixel_values": image,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }
    

# Example usage

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/instructblip-vicuna-7b")   

    dataset = BoneVQADataset(
        csv_file='/dataset/question_bonedata.csv',
        data_file='/dataset/cleaned_output_bonedata.json',
        tokenizer=tokenizer,  # Replace with your tokenizer
        image_transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        max_length=512,
        num_questions=1,
    )

    # Get a sample
    sample = dataset[0]