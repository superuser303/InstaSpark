from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import pandas as pd

# Load captions
df = pd.read_csv("../data/captions.csv")
captions = df["caption"].tolist()

# Save captions to a text file for training
with open("../data/captions.txt", "w") as f:
    for caption in captions:
        f.write(caption + "\n")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Create dataset for language modeling
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="../data/captions.txt",
    block_size=128,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training arguments (disable WandB)
training_args = TrainingArguments(
    output_dir="../models/results",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Increased for better convergence
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    report_to="none",  # Disable WandB logging
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("../models/finetuned_model")