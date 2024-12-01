from typing import TypeVar, Union, List, Dict, Optional
from .base import EmbeddingCache, Reranker
import numpy as np
__all__ = ["Encoder"]


class Encoder(Reranker):
    def __init__(
        self,
        attr: Union[str, List[str]],
        key: str,
        encoder,
        normalize: bool = True,
        k: Optional[int] = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            key=key,
            attr=attr,
            encoder=encoder,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )

    def _encoder(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """Computes embeddings with proper text preparation"""
        text_inputs = []
        for doc in documents:
            # Combine all specified attributes into a single string
            doc_texts = []
            for field in self.attr:
                # Handle both string and list values
                field_value = doc.get(field, "")
                if isinstance(field_value, list):
                    field_value = " ".join(str(v) for v in field_value)
                elif not isinstance(field_value, str):
                    field_value = str(field_value)
                doc_texts.append(field_value)
            text_inputs.append(" ".join(doc_texts))
        
        return self.encoder(text_inputs)

    def __call__(
        self,
        q: Union[List[str], str],
        documents: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        k: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[List[List[Dict[str, str]]], List[Dict[str, str]]]:
        if not documents:
            return [] if isinstance(q, str) else [[]]

        k = k or self.k or len(self)
        
        queries = [q] if isinstance(q, str) else q
        processed_documents = [documents] if isinstance(q, str) else documents
        
        try:
            embeddings_queries = self.encoder(queries)
            rank = self.encode_rank(
                embeddings_queries=embeddings_queries,
                documents=processed_documents,
                k=k,
                batch_size=batch_size or self.batch_size,
            )
            return rank[0] if isinstance(q, str) else rank
        except Exception as e:
            print(f"Encoding error: {e}")
            return [] if isinstance(q, str) else [[]]