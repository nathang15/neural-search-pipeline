import typing
from tqdm import tqdm
from index.faiss_idx import Faiss
from utils.batch import iterate_in_batches
from .base import Retriever

__all__ = ["Encoder"]


class Encoder(Retriever):
    """
    Retrieve documents based on semantic similarity.

    Encodes both queries and documents using a single model, compatible with
    SentenceTransformers or HuggingFace similarity models. Pre-computed document
    embeddings are stored in a Faiss index for fast k-NN searches.
    """

    def __init__(
        self,
        encoder,
        key: str,
        attr: typing.Union[str, typing.List[str]],
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
        index=None,
    ) -> None:
        """
        Initialize the Encoder.

        Args:
            encoder: The model used for encoding queries and documents.
            key (str): Unique key to identify documents in the index.
            attr (Union[str, List[str]]): Attributes to concatenate for encoding.
            normalize (bool): Whether to normalize embeddings.
            k (Optional[int]): Number of neighbors to retrieve.
            batch_size (int): Batch size for encoding.
            index: Pre-built Faiss index (optional).
        """
        super().__init__(key=key, attr=attr, k=k, batch_size=batch_size)
        self.encoder = encoder
        self.index = (
            Faiss(key=self.key, index=index, normalize=normalize)
            if index
            else Faiss(key=self.key, normalize=normalize)
        )

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.index)

    def add(
        self,
        documents: typing.List[typing.Dict[str, str]],
        batch_size: int = 64,
        tqdm_bar: bool = True,
    ) -> "Encoder":
        """
        Add documents to the index.

        Args:
            documents (List[Dict[str, str]]): List of documents to index.
            batch_size (int): Batch size for encoding.
            tqdm_bar (bool): Whether to display a progress bar.

        Returns:
            Encoder: The current instance with updated index.
        """
        for batch in iterate_in_batches(
            sequence=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Indexing",
            tqdm_bar=tqdm_bar,
        ):
            embeddings = self.encoder(
                [
                    " ".join(doc.get(field, "") for field in self.attr)
                    for doc in batch
                ]
            )
            self.index.add(documents=batch, embeddings=embeddings)
        return self

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        tqdm_bar: bool = True,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """
        Retrieve documents from the index based on the query.

        Args:
            q (Union[List[str], str]): Query or list of queries.
            k (Optional[int]): Number of neighbors to retrieve. Defaults to the index size.
            batch_size (Optional[int]): Batch size for query encoding.
            tqdm_bar (bool): Whether to display a progress bar.

        Returns:
            Union[List[List[Dict[str, str]]], List[Dict[str, str]]]: Retrieved documents.
        """
        k = k or len(self)
        batch_size = batch_size or self.batch_size

        results = []
        for batch in iterate_in_batches(
            sequence=[q] if isinstance(q, str) else q,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Retrieving",
            tqdm_bar=tqdm_bar,
        ):
            embeddings = self.encoder(batch)
            results.extend(self.index(k=k, embeddings=embeddings))

        return results[0] if isinstance(q, str) else results
