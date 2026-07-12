import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Railway بيضبط PORT تلقائياً
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",  # لازم يكون 0.0.0.0 عشان Railway يشوفه
        port=port,
        reload=True
    )