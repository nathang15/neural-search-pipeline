# abstract class for ranking models

from .base import Ranker
from .dpr import DPR

__all__ = ["DPR", "Encoder", "Embedding"]
