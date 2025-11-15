from fastapi import FastAPI
from app.controlers import mlControler, openAiControler, explainabilityControler

app = FastAPI(title="Datathon Backend API")

# Include routers
app.include_router(mlControler.router)
app.include_router(openAiControler.router)
app.include_router(explainabilityControler.router)

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    await mlControler.load_model()


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "FastAPI backend is running",
        "model_loaded": mlControler.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)