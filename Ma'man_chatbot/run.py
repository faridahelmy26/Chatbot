import os
import uvicorn

if __name__ == "__main__":
    # Railway بيحدد المنفذ تلقائياً
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",  # لازم يكون 0.0.0.0 عشانRailway يشوفه
        port=port,
        reload=False  # في الإنتاج، خليها False
    )
