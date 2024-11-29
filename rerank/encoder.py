from typing import TypeVar, Union, List, Dict, Optional
from .base import EmbeddingCache, Reranker

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

    def __call__(
        self,
        q: Union[List[str], str],
        documents: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        k: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Union[List[List[Dict[str, str]]], List[Dict[str, str]]]:
        k = k or self.k or len(self)
        
        if not documents:
            return [] if isinstance(q, str) else [[]]

        queries = [q] if isinstance(q, str) else q
        
        embeddings_queries = self.encoder(queries)
        
        processed_documents = [documents] if isinstance(q, str) else documents
        
        rank = self.encode_rank(
            embeddings_queries=embeddings_queries,
            documents=processed_documents,
            k=k,
            batch_size=batch_size or self.batch_size,
        )

        return rank[0] if isinstance(q, str) else rank