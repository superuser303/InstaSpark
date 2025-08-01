from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("Marcuso/instagram-images-with-captions-and-embeddings")

# Extract captions
captions = dataset["train"]["caption"]

# Save to CSV
df = pd.DataFrame({"caption": captions})
df.to_csv("../data/captions.csv", index=False)