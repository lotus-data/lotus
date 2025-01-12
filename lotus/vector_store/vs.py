from abc import ABC, abstractmethod 

import pandas as pd 

class VS(ABC):
    """Abstract class for vector stores."""

    def __init__(self) -> None:
        pass 

    @abstractmethod 
    def index(self, docs: pd.Series, index_dir):
        pass 