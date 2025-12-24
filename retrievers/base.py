from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class RetrieverBase(ABC):
    """Base retriever interface."""

    @abstractmethod
    def search(self, query) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        raise NotImplementedError
