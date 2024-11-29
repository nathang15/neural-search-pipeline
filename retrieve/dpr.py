from typing import List, Dict, Union, Optional

import tqdm

from ..index.faiss_idx import Faiss
from ..utils.batch import iterate_in_batches
from .base import Retriever

class DPR(Retriever):
    """
    Dense Passage Retrieval (DPR) as retriever

    """

    def __init__(
        self,
        key: str,
        attr: Union[str, List[str]],
        encoder,
        query_encoder=None,
        normalize: bool = True,
        k: Optional[int] = None,
        batch_size: int = 128,
        index=None,
    ) -> None:
        """
        Initialize

        Args:
            key (str): identifier for the document index
            attr (Union[str, List[str]]): Document attributes to retrieve on
            encoder: encoder model
            query_encoder
            normalize (bool): Normalize embeddings for cosine similarity
            k (Optional[int]): Default number of top-k retrievals
            batch_size (int): Encoding and retrieval batch size
            index: Faiss index
        """
        super().__init__(key=key, attr=attr, k=k, batch_size=batch_size)
        
        self.encoder = encoder
        self.query_encoder = query_encoder or encoder
        
        self.index = (
            Faiss(key=self.key, index=index, normalize=normalize)
            if index is not None
            else Faiss(key=self.key, normalize=normalize)
        )

    def __len__(self) -> int:
        """Return the current number of indexed documents."""
        return len(self.index)

    def add(
        self,
        documents: List[Dict[str, str]],
        batch_size: Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> 'DPR':
        """
        Add documents to the index

        Args:
            documents (List[Dict[str, str]]): Documents to index.
            batch_size (Optional[int]): Override default batch size.
            tqdm_bar (bool): Display progress bar.

        Returns:
            DPR: Updated retriever instance
        """
        batch_size = batch_size or self.batch_size
        
        # Optimize document text preparation
        def prepare_text(doc: Dict[str, str]) -> str:
            return " ".join(doc.get(field, "").strip() for field in self.attr)
        
        for batch in iterate_in_batches(
            sequence=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Indexing",
            tqdm_bar=tqdm_bar,
        ):
            texts = [prepare_text(doc) for doc in batch]

            embeddings = self.encoder(texts)
            
            self.index.add(documents=batch, embeddings=embeddings)

        self.k = len(self.index)
        return self

    def __call__(
        self,
        q: Union[List[str], str],
        k: Optional[int] = None,
        batch_size: Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> Union[List[List[Dict[str, str]]], List[Dict[str, str]]]:
        """
        Retrieve top-k documents for given query/queries

        Args:
            q (Union[List[str], str]): Query or queries to retrieve against
            k (Optional[int]): Number of top documents to retrieve
            batch_size (Optional[int]): Override default batch size
            tqdm_bar (bool): Display progress bar

        Returns:
            Retrieved documents
        """
        k = k or len(self)
        batch_size = batch_size or self.batch_size

        if isinstance(q, str):
            q = [q]

        results = []
        for batch in iterate_in_batches(
            sequence=q,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} Retrieving",
            tqdm_bar=tqdm_bar,
        ):
            # Encode queries in batches 
            query_embeddings = self.query_encoder(batch)
            
            batch_results = self.index(
                embeddings=query_embeddings,
                k=k,
            )
            results.extend(batch_results)

        return results[0] if len(q) == 1 else results