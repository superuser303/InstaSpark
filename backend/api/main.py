from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the fine-tuned model (after training)
generator = pipeline("text-generation", model="../models/finetuned_model")

@app.get("/")
def read_root():
    return {"message": "Welcome to InstaSpark!"}

@app.post("/generate")
def generate_caption(description: str):
    prompt = f"Write an Instagram caption about {description}: "
    caption = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
    return {"caption": caption}