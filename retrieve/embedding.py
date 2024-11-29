import typing

import numpy as np

from ..index.faiss_idx import Faiss
from ..utils.batch import iterate_in_batches
from .base import Retriever

__all__ = ["Embedding"]


class Embedding(Retriever):
    """
    Retrieve documents based on a custom model or embeddings by the users
    The embeddings must be of shape (n_documents, dim_embeddings)
    Can add embeddings one by one (no need to add all at once)
    """

    def __init__(
        self,
        key: str,
        index=None,
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(key=key, attr="", k=k, batch_size=batch_size)
        self.index = index or Faiss(key=self.key, normalize=normalize)

    def __len__(self) -> int:
        return len(self.index)

    def add(
        self,
        documents: list,
        embeddings_docs: np.ndarray,
        **kwargs,
    ) -> "Embedding":
        """
        Add embeddings both documents and users
        """
        self.index.add(documents=documents, embeddings=embeddings_docs)
        return self


    def __call__(
        self,
        q: np.ndarray,
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """
        Retrieve documents from the index
        """
        k = k or len(self)
        q = q.reshape(-1, q.shape[-1]) if q.ndim == 1 else q

        results = []
        batch_iterator = iterate_in_batches(
            sequence=q,
            batch_size=batch_size or self.batch_size,
            desc=f"{self.__class__.__name__} Retriever",
            tqdm_bar=tqdm_bar,
        )

        results = [
            result 
            for batch in batch_iterator 
            for result in self.index(embeddings=batch, k=k)
        ]

        return results[0] if len(q) == 1 else results