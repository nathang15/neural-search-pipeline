import typing
import tqdm

from ..index import Faiss
from ..utils import yield_batch
from .base import Retriever

__all__ = ["Encoder"]

class Encoder(Retriever):
    # Retreive documnents based on semantic similarity.
    # Encodes both queries and documents within a single model, compatible with sentencetransformers or huggingface similiarity models.
    # Uses Faiss to store pre-computed document embeddings in an index => fast knn search.

    def __init__(
        self,
        encoder,
        key: str,
        attr: typing.Union[str, list],
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
        index = None,
    ) -> None:
        super().__init__(
            key = key,
            attr = attr,
            k = k,
            batch_size = batch_size,
        )
        self.encoder = encoder
        if not index:
            self.index = Faiss(key = self.key, normalize = normalize)
        else:
            self.index = Faiss(key = self.key, index = index, normalize = normalize)

    def __len__(self) -> int:
        return len(self.index)
    
    def add(
        self,
        documents: typing.List[typing.Dict[str, str]],
        batch_size: int = 64,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> "Encoder":
        # add documents to the index

        for batch in yield_batch(
            array = documents,
            batch_size = batch_size,
            desc = f"{self.__class__.__name__} Indexing",
            tqdm_bar = tqdm_bar,
        ):
            self.index.add(
                documents = batch,
                embeddings = self.encoder(
                    [
                        " ".join([doc.get(field, "") for field in self.attr])
                        for doc in batch
                    ]
                ),
            )
        return self
    
    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        # Retrieve k documents from the index from a query or list of queries
        k = k if k else len(self)

        rank = []
        for batch in yield_batch(
            array = q,
            batch_size = batch_size if batch_size else self.batch_size,
            desc = f"{self.__class__.__name__} Retrieving",
            tqdm_bar = tqdm_bar,
        ):
            rank.extend(
                self.index(
                    k = k,
                    embeddings = self.encoder(batch),
                )
            )
        return rank[0] if isinstance(q, str) else rank