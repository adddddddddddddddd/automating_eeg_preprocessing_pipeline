from sqlmodel import SQLModel, Field

class Run(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id")
    status: str
    created_at: str = None
    last_opened_at: str = None
    started_at: str = None
    completed_at: str = None
    

class Dataset(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    description: str
    path : str
    
class Subject(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    dataset_id: int
    subject_id: str
    
class DatasetTasks(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    dataset_id: int
    name: str
    status: str