from sqlmodel import Session, select, create_engine, SQLModel
from models import *

DATABASE_URL = "postgresql://admin:secretsecret@localhost:5432/eeg_pipeline_db"
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Create all database tables"""
    from models import Dataset, Run, Step, AgentLog
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get a database session"""
    return Session(engine)

