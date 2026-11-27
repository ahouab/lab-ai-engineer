from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rag import ALIAS, CHAT_MODEL, EMBED_MODEL, OLLAMA, QDRANT, Question, ServiceError
from rag import answer, request_json, retrieve


app = FastAPI(title="Runbooks du lab AI Engineer", redoc_url=None)


@app.exception_handler(ServiceError)
async def dependency_error(_request, error):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/healthz")
def health():
    return {"status": "up"}


@app.get("/readyz")
def ready():
    models = request_json(f"{OLLAMA}/api/tags", timeout=5)
    names = {model["name"] for model in models["models"]}
    required = {name if ":" in name else f"{name}:latest" for name in (CHAT_MODEL, EMBED_MODEL)}
    missing = required - names
    if missing:
        return JSONResponse(status_code=503, content={"missing_models": sorted(missing)})
    request_json(f"{QDRANT}/collections/{ALIAS}", timeout=5)
    return {"status": "ready", "chat_model": CHAT_MODEL, "embedding_model": EMBED_MODEL}


@app.post("/search")
def search(body: Question):
    return {"sources": retrieve(body.question)}


@app.post("/ask")
def ask(body: Question):
    return answer(body.question)
