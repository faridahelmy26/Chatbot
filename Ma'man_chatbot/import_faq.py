import sys
import os
import json

# أضف المجلد الرئيسي إلى مسار Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import Database
from app.embeddings import get_embedding_engine


def import_faq_data():
    """Import FAQ data from JSON file"""
    db = Database()
    
    # جرب مسارات مختلفة للملف
    json_paths = [
        'data/faq_data.json',
        'faq_data.json',
        '../data/faq_data.json',
    ]
    
    data = None
    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Found file at: {path}")
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("❌ File not found! Please check these locations:")
        for path in json_paths:
            print(f"   - {path}")
        return
    
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
            print(f"⚠️  Skipped (duplicate): {item['question_ar'][:30]}...")
    
    print("\n" + "="*50)
    print(f"📊 Import Results:")
    print(f"   ✅ Added: {added}")
    print(f"   ⚠️  Skipped: {skipped}")
    print(f"   📝 Total: {added + skipped}")
    print("="*50)
    
    if added > 0:
        print("🔄 Rebuilding embeddings...")
        get_embedding_engine().build_embeddings()
        print("✅ Embeddings rebuilt successfully!")
    
    db.close()


if __name__ == "__main__":
    import_faq_data()