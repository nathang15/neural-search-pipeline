import abc
from ..pipeline import Pipeline, And, Or
import os
import pickle

__all__ = ["Reranker"]

class Reranker(abc.ABC):
    """Abstract class for ranking models."""

    def __init__(self, attr: str, encoder, k: int, path: str, similarity) -> None:
        self.attr = attr
        self.encoder = encoder
        self.k = k
        self.path = path
        self.similarity = similarity
        self.embeddings = self.load_embeddings(path = path) if self.path else {}

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__} Ranker"
        repr_str += f"\n  Attributes: {self.attr}"
        repr_str += f"\n  Top-k: {self.k}"
        repr_str += f"\n  Similarity: {self.similarity.__name__}"
        if self.path:
            repr_str += f"\n  Embeddings Path: {self.path}"
        return repr_str

    @abc.abstractmethod
    def __call__(self, documents: list[dict], k: int, attr: str) -> list:
        """
        Abstract method to rank documents.
        
        Args:
            documents (list[dict]): documents input
            k (int): k top results
            attr (str): atrributes to rank on

        Returns:
            list: A list of top k documents
        """
        if not documents:
            return []
        return self
    
    def add(self, documents) -> None:
        """
        Pre-compute embeddings and store them at the path

        Args:
            documents (list[dict]): documents input
        """
        documents = [document[self.attr] for document in documents if document[self.attr] not in self.embeddings]

        if documents:
            for doc, emb in zip(documents, self.encoder.encode(documents)):
                self.embeddings[doc] = emb
            
            if self.path:
                self.dump_embeddings(embeddings = self.embeddings, path = self.path)
        
        return self

    def _rank(self, similarities: list, documents: list):
        """
        Rank documents based on similarities.

        Args:
            similarities (list): list of similarities
            documents (list): list of documents

        Returns:
            list: A list of ranked documents
        """
        top_similarities = sorted(similarities, key=lambda x: -x[1])[:self.k]
        return [
            {**documents[idx], "similarity": score} for idx, score in top_similarities
        ]

    @staticmethod
    def load_embeddings(path: str):
        """
        Load embeddings from a file.

        ### Parameters
        - **path** (`str`): Path to the embeddings file.

        ### Returns
        - (`Dict`): Loaded embeddings.
        """
        if not os.path.isfile(path):
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def dump_embeddings(embeddings, path: str) -> None:
        """
        ### Parameters
        - **embeddings**: Embeddings to save.
        - **path**: Path to save the embeddings.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)

    def __add__(self, other):
        """Pipeline operator."""
        return other + self if isinstance(other, Pipeline) else Pipeline(models=[other, self])

    def __or__(self, other):
        """Or operator."""
        return Or(models=[self, other]) if not isinstance(other, Or) else Or(models=[self] + other.models)

    def __and__(self, other):
        """And operator."""
        return And(models=[self, other]) if not isinstance(other, And) else And(models=[self] + other.models)