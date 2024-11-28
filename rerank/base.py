import abc
from ..pipeline import Pipeline, And, Or
import os
import typing
import numpy as np
import numpy.typing as npt
from utils.batch import iterate_in_batches
import tqdm

__all__ = ["Reranker"]

class EmbeddingCache:
    """
    Store embeddings of rerankers
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.embeddings: typing.Dict[str, npt.NDArray] = {}

    def __len__(self) -> int:
        return len(self.embeddings)

    def add(
        self,
        embeddings: typing.List[npt.NDArray],
        documents: typing.List[typing.Dict[str, str]],
        **kwargs,
    ) -> "EmbeddingCache":
        """
        Pre-compute embeddings and store them
        """
        for document, embedding in zip(documents, embeddings):
            self.embeddings[document[self.key]] = embedding
        return self

    def get(
        self,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        **kwargs,
    ) -> typing.Tuple[typing.List[str], typing.List[npt.NDArray], typing.List[typing.Dict[str, str]]]:
        known: typing.List[str] = []
        embeddings: typing.List[npt.NDArray] = []
        unknown: typing.List[typing.Dict[str, str]] = []

        for batch in documents:
            for document in batch:
                key = document[self.key]
                if key in self.embeddings:
                    known.append(key)
                    embeddings.append(self.embeddings[key])
                else:
                    unknown.append(document)
        
        return known, embeddings, unknown
    
    def clear(self) -> None:
        """
        Clear all stored embeddings.
        """
        self.embeddings.clear()

    def contains(self, key: str) -> bool:
        return key in self.embeddings

class Reranker(abc.ABC):
    """Abstract class for ranking models."""

    def __init__(
            self, 
            key: str,
            attr: typing.Union[str, typing.List[str]], 
            encoder, 
            normalize: bool,
            batch_size: int,
            k: typing.Optional[int] = None
        ) -> None:
        self.key = key,
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
        q: typing.Union[typing.List[str], str], 
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: int,
        batch_size: typing.Optional[int] = None,
        **kwargs,) -> typing.Union[typing.List[typing.List[typing.Dict[str, str]]], typing.List[typing.Dict[str, str]]]:
        """
        rerank documents based on query
        """
        if isinstance(q, str):
            return []
        elif isinstance(q, list):
            return [[]]
    
    def _encoder(self, documents: typing.List[typing.Dict[str, str]]) -> np.ndarray:
        """computes embeddings"""
        return self.encoder(
            [
                " ".join([doc.get(field, "") for field in self.attr]) for doc in documents
            ]
        )
    
    def _encode(
        self,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        batch_size: typing.Optional[int] = None,
    ) -> typing.Dict[str, np.ndarray]:
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
    
    def _batch_encode(self, documents: typing.List[typing.Dict[str, str]], batch_size: int, desc: str) -> typing.List[np.ndarray]:
        """computes embeddings in batches"""
        embeddings = []
        for batch in iterate_in_batches(
            sequence=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Reranker",
        ):
            embeddings.extend(self._encoder(documents = batch))
        return embeddings
    
    def add(self, documents: typing.List[typing.Dict[str, str]], batch_size: int = 64) -> "Reranker":
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
        embeddings_documents: typing.Dict[str, np.ndarray],
        embeddings_queries: np.ndarray,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        k: int,
        batch_size: typing.Optional[int] = None,
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
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        k: int,
        batch_size: typing.Optional[int] = None,
    ) -> typing.List[typing.List[typing.Dict[str, str]]]:
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