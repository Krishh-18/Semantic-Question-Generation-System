from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
dataset = load_dataset("squad")

# -----------------------------
# 2. TF-IDF Setup (on contexts)
# -----------------------------
contexts = [example["context"] for example in dataset["train"].select(range(5000))]
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(contexts)

# -----------------------------
# 3. Load Model + Tokenizer
# -----------------------------
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# -----------------------------
# 4. Preprocessing Function (with TF-IDF filtering)
# -----------------------------
def preprocess(example, idx):
    context = example["context"]

    # TF-IDF score for this context (importance)
    tfidf_vector = tfidf.transform([context])
    score = np.sum(tfidf_vector.toarray())

    # Keep only informative contexts (simple threshold)
    if score < 1.0:
        context = context[:200]  # fallback: truncate low-info text

    input_text = "generate question: " + context
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
# 5. Apply Preprocessing
# -----------------------------
dataset = dataset["train"].select(range(5000))
dataset = dataset.map(preprocess, with_indices=True)

# -----------------------------
# 6. Data Collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------
# 7. Training Arguments
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
# 8. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# -----------------------------
# 9. Train Model
# -----------------------------
trainer.train()

# -----------------------------
# 10. Save Model
# -----------------------------
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("✅ Training complete with TF-IDF enhancement.")