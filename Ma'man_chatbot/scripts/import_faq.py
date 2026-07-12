import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.models.database import Database
from app.core.embeddings import get_embedding_engine

def main():
    print("🚀 Importing FAQ data...")
    
    db = Database()
    
    # Check if file exists
    json_file = project_root / "data" / "faq_data.json"
    if not json_file.exists():
        print(f"❌ File not found: {json_file}")
        print("📁 Please create data/faq_data.json")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    added = 0
    skipped = 0
    
    for item in data:
        success = db.insert_faq(
            question_ar=item.get('question_ar', '').strip(),
            answer_ar=item.get('answer_ar', '').strip(),
            question_en=item.get('question_en', '').strip(),
            answer_en=item.get('answer_en', '').strip(),
            category=item.get('category', 'General')
        )
        if success:
            added += 1
            print(f"✅ Added: {item['question_ar'][:30]}...")
        else:
            skipped += 1
    
    print(f"\n✅ Added {added} FAQs (Skipped {skipped} duplicates)")
    
    if added > 0:
        print("🔄 Building embeddings...")
        try:
            get_embedding_engine().build_embeddings()
            print("✅ Done!")
        except Exception as e:
            print(f"⚠️  Embeddings build error: {e}")
    
    db.close()

if __name__ == "__main__":
    main()
