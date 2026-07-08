from sentence_transformers import SentenceTransformer
from pathlib import Path

print("🔄 Downloading model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Save model locally
model_path = Path("models/paraphrase-multilingual-MiniLM-L12-v2")
model_path.parent.mkdir(exist_ok=True)
model.save(str(model_path))

print(f"✅ Model saved successfully to: {model_path}")
