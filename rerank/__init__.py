# abstract class for reranking models

from .base import Reranker
from .dpr import DPR

__all__ = ["DPR", "Encoder", "Embedding"]
