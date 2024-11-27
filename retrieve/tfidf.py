import typing
import os
import numpy as np
from scipy.sparse import csr_matrix, hstack
from functools import lru_cache

from lenlp import sparse
from utils.batch import iterate_in_batches
from .base import Retriever

__all__ = ["TfIdf"]

class TfIdf(Retriever):
    def __init__(
        self,
        key: str,
        attr: typing.Union[str, list],
        documents: typing.List[typing.Dict[str, str]] = None,
        tfidf: sparse.TfidfVectorizer = None,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
        fit: bool = True,
        max_workers: int = None
    ) -> None:
        super().__init__(key=key, attr=attr, k=k, batch_size=batch_size)

        self.tfidf = (
            sparse.TfidfVectorizer(
                normalize=True, 
                ngram_range=(3, 7), 
                analyzer="char_wb"
            )
            if tfidf is None else tfidf
        )

        self.max_workers = max_workers or (os.cpu_count() or 1)

        self.documents = []
        self.duplicates = set()

        if documents:
            self._process_initial_documents(documents, fit)

    def _process_initial_documents(self, documents, fit=True):
        # Filter out duplicates
        unique_docs = [
            doc for doc in documents 
            if doc[self.key] not in self.duplicates
        ]

        texts = [
            " ".join([doc.get(field, "") for field in self.attr]) 
            for doc in unique_docs
        ]

        method = self.tfidf.fit_transform if fit else self.tfidf.transform
        self.matrix = csr_matrix(
            method(texts),
            dtype=np.float32
        ).T

        for doc in unique_docs:
            self.documents.append({self.key: doc[self.key]})
            self.duplicates.add(doc[self.key])

        self.n = len(self.documents)
        self.k = len(self.documents) if self.k is None else self.k

    @lru_cache(maxsize=1000)
    def _cached_transform(self, text):
        """Cached TF-IDF transformation"""
        return self.tfidf.transform([text])

    def add(
        self,
        documents: list,
        batch_size: int = 100_000,
        tqdm_bar: bool = False,
        **kwargs
    ):
        for batch in iterate_in_batches(
            documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc="Adding documents to TfIdf retriever",
        ):
            batch = [
                document
                for document in batch
                if document[self.key] not in self.duplicates
            ]

            if not batch:
                continue

            sp_matrix = csr_matrix(
                [
                    self._cached_transform(" ".join([doc.get(field, "") for field in self.attr])).toarray()[0]
                    for doc in batch
                ],
                dtype=np.float32,
            ).T

            self.matrix = hstack((self.matrix, sp_matrix))

            for document in batch:
                self.documents.append({self.key: document[self.key]})
                self.duplicates[document[self.key]] = True

            self.n += len(batch)

        return self

    def top_k(self, similarities, k):
        matchs, scores = [], []
        for row in similarities:
            _k = min(row.data.shape[0] - 1, k)
            ind = np.argpartition(row.data, kth=_k, axis=0)[:k]
            similarity = np.take_along_axis(row.data, ind, axis=0)
            indices = np.take_along_axis(row.indices, ind, axis=0)
            ind = np.argsort(similarity, axis=0)
            scores.append(-1 * np.take_along_axis(similarity, ind, axis=0))
            matchs.append(np.take_along_axis(indices, ind, axis=0))
        return matchs, scores

    def __call__(
        self,
        q: typing.Union[str, typing.List[str]],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        k = k if k is not None else self.k

        results = []

        for batch in iterate_in_batches(
            sequence=q,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            similarities = -1 * csr_matrix([self._cached_transform(text).toarray()[0] for text in batch]).dot(self.matrix)

            batch_match, batch_similarities = self.top_k(similarities, k)

            for match, similarities in zip(batch_match, batch_similarities):
                results.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return results[0] if isinstance(q, str) else results