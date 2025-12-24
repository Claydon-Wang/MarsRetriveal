from abc import ABC, abstractmethod
from typing import Dict


class EvaluatorBase(ABC):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(self, pred_df, label: str = "run") -> Dict:
        raise NotImplementedError
