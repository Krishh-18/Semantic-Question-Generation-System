from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
dataset = load_dataset("squad")

# -----------------------------
# 2. Load Model + Tokenizer
# -----------------------------
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# -----------------------------
# 3. Preprocessing Function
# ----------------------------- 
def preprocess(example):
    input_text = "generate question: " + example["context"]
    target_text = example["question"]

    model_inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        target_text,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# -----------------------------
# 4. Apply Preprocessing
# -----------------------------
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# -----------------------------
# 5. Data Collator (handles padding properly)
# -----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------
# 6. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=3e-5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2
)

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(5000)), 
    data_collator=data_collator,
)

# -----------------------------
# 8. Train Model
# -----------------------------
trainer.train()

# -----------------------------
# 9. Save Model
# -----------------------------
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("✅ Training complete. Model saved as 'fine_tuned_model'")