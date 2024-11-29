import abc
from ..pipeline import Pipeline, And, Or
import os
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..utils.batch import iterate_in_batches
import tqdm

__all__ = ["Reranker"]

class EmbeddingCache:
    """
    Store embeddings of rerankers
    """

    def __init__(
        self, 
        key: str, 
        cache_dir: Optional[str] = None, 
        max_memory_size: int = 10000
    ):
        self.key = key
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings  = {}
        self.max_memory_size = max_memory_size

    def _get_cache_path(self, identifier: str) -> str:
        return os.path.join(self.cache_dir, f"{identifier}.npy")

    def add(
        self, 
        embeddings: List[npt.NDArray], 
        documents: List[Dict[str, str]],
        **kwargs,
    ) -> 'EmbeddingCache':
        for document, embedding in zip(documents, embeddings):
            key = document[self.key]
            
            if len(self.embeddings) >= self.max_memory_size:
                oldest_key = next(iter(self.embeddings))
                del self.embeddings[oldest_key]
            
            self.embeddings[key] = embedding
            np.save(self._get_cache_path(key), embedding)
        
        return self

    def get(
        self, 
        documents: List[List[Dict[str, str]]]
    ) -> Tuple[List[str], List[npt.NDArray], List[Dict[str, str]]]:
        known, embeddings, unknown = [], [], []

        for batch in documents:
            for document in batch:
                key = document[self.key]
                
                if key in self.embeddings:
                    known.append(key)
                    embeddings.append(self.embeddings[key])
                    continue
                
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    embedding = np.load(cache_path)
                    known.append(key)
                    embeddings.append(embedding)
                    self.embeddings[key] = embedding
                else:
                    unknown.append(document)
        
        return known, embeddings, unknown

class Reranker(abc.ABC):
    """Abstract class for ranking models."""

    def __init__(
        self, 
        key: str,
        attr: Union[str, List[str]], 
        encoder, 
        normalize: bool,
        batch_size: int,
        k: Optional[int] = None
        ) -> None:
        self.key = key
        self.attr = attr if isinstance(attr, list) else [attr]
        self.encoder = encoder
        self.store = EmbeddingCache(key=self.key)
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__} Reranker"
        repr_str += f"\n  Key: {self.key}"
        repr_str += f"\n  Attributes: {self.attr}"
        repr_str += f"\n  Normalize: {self.normalize}"
        repr_str += f"\n  Embeddings: {len(self.store)}"
        return repr_str

    @abc.abstractmethod
    def __call__(
        self,
        q: Union[List[str], str], 
        documents: Union[
            List[List[Dict[str, str]]],
            List[Dict[str, str]],
        ],
        k: int,
        batch_size: Optional[int] = None,
        **kwargs,) -> Union[List[List[Dict[str, str]]], List[Dict[str, str]]]:
        """
        rerank documents based on query
        """
        if isinstance(q, str):
            return []
        elif isinstance(q, list):
            return [[]]
    
    def _encoder(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """computes embeddings"""
        return self.encoder(
            [
                " ".join([doc.get(field, "") for field in self.attr]) for doc in documents
            ]
        )
    
    def _encode(
        self,
        documents: List[List[Dict[str, str]]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes documents embeddings if not yet done"""
        final, embeddings, missed = self.store.get(documents=documents)
        if missed:
            missed_embeddings = self._batch_encode(
                documents=missed,
                batch_size=batch_size,
                desc=f"{self.__class__.__name__} Indexing missed documents",
            )

            # Merge
            final += [document[self.key] for document in missed]
            embeddings = embeddings + missed_embeddings

        return {key: embedding for key, embedding in zip(final, embeddings)}
    
    def _batch_encode(self, documents: List[Dict[str, str]], batch_size: int, desc: str) -> List[np.ndarray]:
        """computes embeddings in batches"""
        embeddings = []
        for batch in iterate_in_batches(
            sequence=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Reranker",
        ):
            embeddings.extend(self._encoder(documents = batch))
        return embeddings
    
    def add(self, documents: List[Dict[str, str]], batch_size: int = 64) -> "Reranker":
        """
        Pre-compute embeddings and store them at the path
        """
        self.store.add(
            documents = documents,
            embeddings=self._batch_encode(
                documents=documents,
                batch_size=batch_size,
                desc=f"{self.__class__.__name__} Indexing",
            ),
        )
        return self

    def rank(
        self,
        embeddings_documents: Dict[str, np.ndarray],
        embeddings_queries: np.ndarray,
        documents: List[List[Dict[str, str]]],
        k: int,
        batch_size: Optional[int] = None,
    ) -> list: 
        """
        Rank documents based on similarities among returned top k
        """
        # Reshape if needed
        if len(embeddings_queries.shape) == 1:
            embeddings_queries = embeddings_queries.reshape(1, -1)

        # Normalize embeddings for cosine similarity
        if self.normalize:
            embeddings_queries = (
                embeddings_queries
                / np.linalg.norm(embeddings_queries, axis=-1)[:, None]
            )

        # Compute sim scores
        scores, missing = [], []
        for q, batch in tqdm.tqdm(
            zip(embeddings_queries, documents), position=0, desc="Ranker scoring"
        ):
            if batch:
                scores.append(
                    q
                    @ np.stack(
                        [embeddings_documents[d[self.key]] for d in batch], axis=0
                    ).T
                )
                missing.append(False)
            else:
                # Did not find any doc for the query
                scores.append(np.array([]))
                missing.append(True)

        ranked = []
        for scores_query, documents_query, missing_query in tqdm.tqdm(
            zip(scores, documents, missing), position=0, desc="Reranking"
        ):
            if missing_query:
                ranked.append([])
                continue

            scores_query = scores_query.reshape(1, -1)
            ranks_query = np.fliplr(np.argsort(scores_query))
            scores_query, ranks_query = scores_query.flatten(), ranks_query.flatten()
            ranks_query = ranks_query[:k]
            ranked.append(
                [
                    {
                        **document,
                        "Similarity": similarity,
                    }
                    for document, similarity in zip(
                        np.take(documents_query, ranks_query),
                        np.take(scores_query, ranks_query),
                    )
                ]
            )

        return ranked
    
    def encode_rank(
        self,
        embeddings_queries: np.ndarray,
        documents: List[List[Dict[str, str]]],
        k: int,
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
        """ Encoder documents and rerank based on the query """
        embeddings_documents = self._encode(documents=documents, batch_size=batch_size)
        return self.rank(
            embeddings_documents=embeddings_documents,
            embeddings_queries=embeddings_queries,
            documents=documents,
            k=k,
            batch_size=batch_size,
        )

    def __add__(self, other):
        """Pipeline operator."""
        return other + self if isinstance(other, Pipeline) else Pipeline(models=[other, self])

    def __or__(self, other):
        """Or operator."""
        return Or(models=[self, other]) if not isinstance(other, Or) else Or(models=[self] + other.models)

    def __and__(self, other):
        """And operator."""
        return And(models=[self, other]) if not isinstance(other, And) else And(models=[self] + other.models)