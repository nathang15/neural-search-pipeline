import abc

__all__ = ["Ranker"]

class Ranker(abc.ABC):
    """Abstract class for ranking models."""

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

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
        pass
