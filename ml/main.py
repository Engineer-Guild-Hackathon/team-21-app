# MLサービス用のメインファイル
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="ML Service", version="1.0.0")


@app.get("/")
async def root():
    return {"message": "ML Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
