from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from lotus.types import RMOutput


class VS(ABC):
    """Abstract class for vector stores."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def index(self, docs: pd.Series, index_dir):
        pass

    @abstractmethod 
    def search(self,
    queries: pd.Series | str | Image.Image | list | NDArray[np.float64],
    K:int,
    **kwargs: dict[str, Any],
 ) -> RMOutput:
        pass 
    
    @abstractmethod
    def get_vectors_from_index(self, collection_name:str, ids: list[int]) -> NDArray[np.float64]:
        pass 