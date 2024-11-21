import abc

from ..pipeline import Intersection, Pipeline, Union

__all__ = ["Retriever"]

class Retriever(abc.ABC):

    def __init__(self, attr: str, k: int) -> None:
        """
        Initialize the retriever with a target attribute

        Args:
            on (str): The attributes that affect retrieval
            documents (list, optional): List of documents
        """
        super().__init__()
        self.attr = attr
        self.k = k
        self.documents = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} retriever"
            f"\n \t attrs: {self.attr}"
            f"\n \t documents: {len(self)}"
        )

    @abc.abstractmethod
    def __call__(self, **kwargs) -> list:
        """
        Abstract method to retrieve documents

        Args:
            q (list): Queries
            k (int, optional): top k results
        """
        pass

    @abc.abstractmethod
    def add(self, documents: list) -> "Retriever":
        """
        Abstract method to add documents to the retriever

        Args:
            documents (list): List of documents to add.
        """
        return self

    def __len__(self) -> int:
        """
        Get the number of documents in the retriever

        """
        return len(self.documents)
    
    def __add__(self, other):
        """ Pipeline ops """
        if isinstance(other, Pipeline):
            return Pipeline(self, other.models)
        return Pipeline([self, other])

    def __or__(self, other):
        """ Union ops """
        if isinstance(other, Union):
            return Union([self] + other.models)
        return Union([self, other])

    def __and__(self, other):
        """ Intersection ops """
        if isinstance(other, Intersection):
            return Intersection([self] + other.models)
        return Intersection([self, other])