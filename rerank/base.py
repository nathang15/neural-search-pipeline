import abc
from ..pipeline import Pipeline, And, Or
import os
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..utils.batch import iterate_in_batches
import tqdm
import hashlib
__all__ = ["Reranker"]

class EmbeddingCache:
    """Store embeddings of rerankers with safe file handling."""

    def __init__(
        self, 
        key: str, 
        cache_dir: Optional[str] = None, 
        max_memory_size: int = 10000
    ):
        self.key = key
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings = {}
        self.max_memory_size = max_memory_size

    def _create_safe_filename(self, identifier: str) -> str:
        """Create a safe filename from any identifier string."""
        # Create a hash of the identifier to ensure a safe, unique filename
        hash_object = hashlib.md5(identifier.encode())
        return hash_object.hexdigest() + '.npy'

    def _get_cache_path(self, identifier: str) -> str:
        """Get the safe cache path for an identifier."""
        safe_filename = self._create_safe_filename(identifier)
        return os.path.join(self.cache_dir, safe_filename)

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
            cache_path = self._get_cache_path(key)
            np.save(cache_path, embedding)
        
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
        if len(embeddings_queries.shape) == 1:
            embeddings_queries = embeddings_queries.reshape(1, -1)

        if self.normalize:
            embeddings_queries = embeddings_queries / (np.linalg.norm(embeddings_queries, axis=-1)[:, None] + 1e-8)

        scores, missing = [], []
        for q, batch in tqdm.tqdm(
            zip(embeddings_queries, documents), position=0, desc="Ranker scoring"
        ):
            if batch:
                doc_embeddings = np.stack([embeddings_documents[d[self.key]] for d in batch], axis=0)
                
                if self.normalize:
                    doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
                
                similarity_scores = np.clip(q @ doc_embeddings.T, -1.0, 1.0)
                scores.append(similarity_scores)
                missing.append(False)
            else:
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
            
            temperature = 0.5
            exp_scores = np.exp(scores_query / temperature)
            softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
            
            ranks_query = np.fliplr(np.argsort(softmax_scores))
            scores_query, ranks_query = softmax_scores.flatten(), ranks_query.flatten()
            ranks_query = ranks_query[:k]

            ranked_docs = []
            for document, similarity in zip(
                np.take(documents_query, ranks_query),
                np.take(scores_query, ranks_query),
            ):
                ranked_doc = document.copy()
                ranked_doc["similarity"] = float(similarity)
                ranked_docs.append(ranked_doc)
            
            # Filter out low confidence matches
            confidence_threshold = 0.01
            ranked_docs = [doc for doc in ranked_docs if doc["similarity"] > confidence_threshold]
            
            ranked.append(ranked_docs)

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