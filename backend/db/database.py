from sqlmodel import SQLModel, create_engine, Session

sql_url = f"sqlite:///model_inference_log.db"

engine = create_engine(sql_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
