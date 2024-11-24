import typing

import numpy as np

from utils.batch import iterate_in_batches

__all__ = ["Faiss"]

class Faiss:
    # key: identifier for each doc
    # index: faiss index

    def __init__(self, key, index = None, normalize: bool = True) -> None:
        import faiss
        self.key = key
        self.index = index
        self.normalize = normalize
        self.documents = []

    def __len__(self) -> int:
        return len(self.documents)
    
    def _build(self, embeddings: np.ndarray):
        # internal used method to build faiss index

        if not self.index:
            try:
                import faiss

                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
            except:
                raise ImportError("Please install faiss to enable the faiss backend.")
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        if not self.index.is_trained and embeddings:
            self.index.train(embeddings)
        self.index.add(embeddings)
        return self.index
    
    def add(self, documents: list, embeddings: np.ndarray) -> "Faiss":
        """Add documents to the faiss index and export embeddings if the path is provided.
        Streaming friendly.

        documents: List of documents as json or list of string to pre-compute queries embeddings.

        """
        array = []
        for document, embedding in zip(documents, embeddings):
            self.documents.append({self.key: document[self.key]})
            array.append(embedding)

        embeddings = np.array(array)
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, None]
        self.index = self._build(embeddings=embeddings)
        return self

    
    def __call__(
        self,
        embeddings: np.ndarray,
        k: typing.Optional[int] = None,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, typing.Any]]],
        typing.List[typing.Dict[str, typing.Any]],
    ]:
        if not k:
            k = len(self)
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, None]
        
        distances, indices = self.index.search(embeddings, k)

        # filter out -1
        matches = np.take(self.documents, np.where(indices < 0, 0, indices))

        rank = []
        for distance, index, match in zip(distances, indices, matches):
            rank.append(
                [
                    {
                        **m,
                        "similarity": 1 / (1 + d),
                    }
                    for d, idx, m in zip(distance, index, match)
                    if idx > -1
                ]
            )
        
        return rank