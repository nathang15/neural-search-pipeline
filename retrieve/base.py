import abc

__all__ = ["Retriever"]

class Retriever(abc.ABC):
    """Retriever base class."""

    def __init__(self, attr: str) -> None:
        """
        Initialize the retriever with a target attribute

        Args:
            on (str): The attributes that affect retrieval
            documents (list, optional): List of documents
        """
        super().__init__()
        self.attr = attr
        self.documents = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} retriever"
            f"\n \t o: {self.attr}"
            f"\n \t documents: {len(self)}"
        )

    @abc.abstractmethod
    def __call__(self, q: list, k: int = None) -> list:
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
        pass

    def __len__(self) -> int:
        """
        Get the number of documents in the retriever

        """
        return len(self.documents)
