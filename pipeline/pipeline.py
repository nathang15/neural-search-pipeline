__all__ = ["Make"]

import collections
import typing

from scipy.special import softmax

from .base import Make, rank_intersection, rank_union, rank_vote

class PipelineUnion(Make):
    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Union Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

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
        ranked = rank_union(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        return PipelineUnion(models=self.models + [other])

    def __and__(self, other) -> "PipelineIntersection":
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> "PipelineVote":
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        if isinstance(other, list):
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])
    
class PipelineIntersection(Make):
    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Intersection Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
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
        scores, counter = self._scores(match=match)
        ranked = rank_intersection(
            key=self.key,
            models=self.models,
            match=match,
            scores=scores,
            counter=counter,
        )
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        return PipelineUnion(models=[self, other])

    def __and__(self, model) -> "PipelineIntersection":
        return PipelineIntersection(models=self.models + [model])

    def __mul__(self, other) -> "PipelineVote":
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        if isinstance(other, list):
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])

class PipelineVote(Make):
    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Voting Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
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
        scores, counter = self._scores(match=match)
        ranked = rank_vote(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> "PipelineIntersection":
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> "PipelineVote":
        return PipelineVote(models=self.models + [other])

    def __add__(self, other) -> "Pipeline":
        if isinstance(other, list):
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])


class Pipeline(Make):
    def __init__(self, models: list) -> None:
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = {**kwargs, "documents": documents}

        for model in self.models:
            if isinstance(model, dict):
                answer = self._map_documents(documents=model, answer=query["documents"])
            else:
                answer = model(q=q, k=k, batch_size=batch_size, **query)

            query.update({"documents": answer})

        return query["documents"]

    def _map_documents(
        self,
        documents: typing.Dict[str, typing.Dict[str, str]],
        answer: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        if not answer:
            return answer

        batch = True
        if isinstance(answer[0], dict):
            batch = False
            answer = [answer]

        mapping = []
        for query_document in answer:
            mapping.append(
                [
                    {
                        **documents.get(document[self.key], {}),
                        self.key: document[self.key],
                        "similarity": document["similarity"],
                    }
                    for document in query_document
                ]
            )

        return mapping[0] if not batch else mapping

    def __or__(self, other) -> PipelineUnion:
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> PipelineIntersection:
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> PipelineVote:
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        if isinstance(other, list):
            return Pipeline(
                models=self.models
                + [{document[self.key]: document for document in other}]
            )
        return Pipeline(models=self.models + [other])