import collections
import typing
from .base import Make, rank_intersection, rank_union, rank_vote
from .pipeline import (Pipeline, PipelineIntersection, PipelineUnion,
                       PipelineVote)


__all__ = ["And", "Or"]

class AndOrVote(Make):
    """ Base class for And and Or ops"""

    def __repr__(self) -> str:
        repr = self.__class__.__name__
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, list):
            return Pipeline(
                [self, {document[self.models[0].key]: document for document in other}]
            )
        return Pipeline([self, other])

    def __or__(self, other) -> PipelineUnion:
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> PipelineIntersection:
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> PipelineVote:
        """Custom operator for voting."""
        return PipelineVote(models=[self, other])



class Or(AndOrVote):
    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = self._build_query(q=q, batch_size=batch_size, k=k, documents=documents)
        match = self._build_match(query=query)
        scores, _ = self._scores(match=match)
        ranked = rank_union(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "Or":
        """Union operator"""
        return Or(models=self.models + [other])


class And(AndOrVote):

    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = self._build_query(q=q, batch_size=batch_size, k=k, documents=documents)
        match = self._build_match(query=query)
        scores, counter = self._scores(match=match)
        ranked = rank_intersection(
            key=self.key,
            models=self.models,
            match=match,
            scores=scores,
            counter=counter,
        )
        return ranked[0] if isinstance(q, str) else ranked

    def __and__(self, other) -> "And":
        return And(models=self.models + [other])
    
class Vote(AndOrVote):
    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = self._build_query(
            q=q, batch_size=batch_size, k=k, documents=documents, **kwargs
        )
        match = self._build_match(query=query)
        scores, _ = self._scores(match=match)
        ranked = rank_vote(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __mul__(self, other) -> "Vote":
        return Vote(models=self.models + [other])