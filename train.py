from model_arch import get_model_
from dataset import BoneVQADataset
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from tqdm import tqdm
import os


# Function to calculate BLEU score
def calculate_BLEU(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        score = sentence_bleu([ref], pred)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0


# Function to calculate BERTScore (F1)
def calculate_BERT_f1(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang='vi', verbose=False)
    return F1.mean().item()


if __name__ == "__main__":
    data_file = 'dataset/cleaned_output_bonedata.json'
    csv_file = 'dataset/question_bonedata.csv'

    model, processor, image_transform, device = get_model_()
    tokenizer = processor.tokenizer

    # 🧠 Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",    
        task_type=TaskType.SEQ_2_SEQ_LM  # Hoặc TaskType.SEQ_2_SEQ_LM tùy mô hình
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset & Dataloader
    dataset = BoneVQADataset(
        csv_file=csv_file,
        data_file=data_file,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=512,
        num_questions=1
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Hyperparameters
    num_epochs = 15
    learning_rate = 5e-5
    total_steps = len(train_loader) * num_epochs
    patience = 3

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler('cuda')
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"🔥 Average training loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        predictions, references = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                pixel_values = batch["pixel_values"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values
                )
                val_loss += outputs.loss.item()

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=64,

                )
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

                predictions.extend(decoded_preds)
                references.extend(decoded_refs)

        avg_val_loss = val_loss / len(val_loader)
        bleu = calculate_BLEU(predictions, references)
        bert_f1 = calculate_BERT_f1(predictions, references)

        print(f"🧪 Val Loss: {avg_val_loss:.4f} | BLEU: {bleu:.4f} | BERT-F1: {bert_f1:.4f}")

        # --- Save model based on val_loss ---
        output_dir = "checkpoints_lora"
        os.makedirs(output_dir, exist_ok=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            save_path = f"{output_dir}/epoch_{epoch+1}_valloss_{avg_val_loss:.4f}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ Loss improved. Model saved at: {save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ Val loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

        model.train()
        torch.cuda.empty_cache()
