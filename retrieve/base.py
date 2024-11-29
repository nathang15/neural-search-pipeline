import abc

from ..pipeline import And, Pipeline, Or
import typing

__all__ = ["Retriever"]

class Retriever(abc.ABC):

    def __init__(
        self,
        key: str,
        attr: typing.Union[str, list], 
        k: typing.Optional[int],
        batch_size: int,
    ) -> None:
        """
        Initialize the retriever with a target attribute

        Args:
            attr (str): The attributes that affect retrieval
            documents (list, optional): List of documents
        """
        super().__init__()
        self.key = key
        self.attr = attr if isinstance(attr, list) else [attr]
        self.documents = None
        self.k = k
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} retriever"
            f"\n \t key: {self.key}"
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
    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int],
        batch_size: typing.Optional[int],
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from the index"""
        return []

    def __len__(self) -> int:
        """
        Get the number of documents in the retriever

        """
        return len(self.documents) if self.documents else 0
    
    def __add__(self, other):
        """ Pipeline ops """
        if isinstance(other, Pipeline):
            return Pipeline(self, other.models)
        elif isinstance(other, list):
            # Documents are part of the pipeline
            return Pipeline(
                [self, {document[self.key]: document for document in other}]
            )
        return Pipeline([self, other])

    def __or__(self, other):
        """ Or ops """
        if isinstance(other, Or):
            return Or([self] + other.models)
        return Or([self, other])

    def __and__(self, other):
        """ And ops """
        if isinstance(other, And):
            return And([self] + other.models)
        return And([self, other])