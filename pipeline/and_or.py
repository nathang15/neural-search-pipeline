import collections

from .base import Make
from .pipeline import Pipeline

__all__ = ["And", "Or"]

class AndOr(Make):
    """ Base class for And and Or ops"""

    def __repr__(self) -> str:
        repr = self.__class__.__name__
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __add__(self, other):
        """Pipeline operator"""
        if isinstance(other, Pipeline):
            return Pipeline([self] + other.models)
        return Pipeline([self, other])


class Or(AndOr):
    """
    Gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers.

    Parameters
        model: List of models
    """

    def __init__(self, models: list):
        self.models = models

    def __call__(self, q: str, **kwargs):
        query = {"q": q, **kwargs}
        documents = []
        for model in self.models:
            for document in model(**query):
                if "similarity" in document:
                    document.pop("similarity")
                # Drop duplicates
                if document in documents:
                    continue
                documents.append(document)
        return documents

    def __or__(self, model):
        self.models.append(model)
        return self


class And(AndOr):
    """
    Gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers only if they intersect

    Parameters
        model: List of models
    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(self, q: str, **kwargs):
        query = {"q": q, **kwargs}
        counter_docs = collections.defaultdict(int)
        for model in self.models:
            for document in model(**query):
                if "similarity" in document:
                    document.pop("similarity")
                counter_docs[tuple(sorted(document.items()))] += 1
        return [
            dict(document) for document, count in counter_docs.items() if count >= len(self.models)
        ]

    def __and__(self, model):
        self.models.append(model)
        return self