from sentence_transformers import SentenceTransformer
from pathlib import Path

print("🔄 Downloading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save model locally
model_path = Path("models/all-MiniLM-L6-v2")
model_path.parent.mkdir(exist_ok=True)
model.save(str(model_path))

print(f"✅ Model saved successfully to: {model_path}")
