from fastapi import FastAPI

from backend import config
from backend.db.database import create_db_and_tables
from backend.endpoints import diagnose, report

app = FastAPI(
    title=config.APP_NAME, description=config.APP_DESCRIPTION 
)

@app.on_event("startup")
async def startup_event():
    create_db_and_tables()

app.include_router(diagnose.router, tags=["inference"])
app.include_router(report.router, tags=["report"])

@app.get("/")
async def root():
    return {"message": config.APP_NAME+" is live and receiving requests...."}
