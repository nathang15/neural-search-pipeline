from __future__ import annotations

import typing
import numpy as np
from typing import List, Dict, Union, Optional

from .base import EmbeddingCache, Reranker

class Embedding(Reranker):
    def __init__(
        self,
        key: str,
        normalize: bool = True,
        k: Optional[int] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(
            key=key,
            attr="",
            encoder=None,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )

    def add(
        self,
        documents: List,
        embeddings_documents: List[np.ndarray],
        **kwargs,
    ) -> "Embedding":
        self.store.add(
            documents=documents,
            embeddings=embeddings_documents,
        )
        return self

    def __call__(
        self,
        q: np.ndarray,
        documents: Union[
            List[List[Dict[str, str]]],
            List[Dict[str, str]],
        ],
        k: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[
        List[List[Dict[str, str]]],
        List[Dict[str, str]],
    ]:
        k = k or self.k or len(self)
        
        is_single_query = q.ndim == 1
        q = np.atleast_2d(q)
        
        documents = [documents] if isinstance(documents, dict) else documents

        known, embeddings, _ = self.store.get(documents=documents)
        embeddings_documents_dict = dict(zip(known, embeddings))

        ranked = self.rank(
            embeddings_queries=q,
            embeddings_documents=embeddings_documents_dict,
            documents=documents,
            k=k,
            batch_size=batch_size or self.batch_size,
        )

        return ranked[0] if is_single_query else ranked