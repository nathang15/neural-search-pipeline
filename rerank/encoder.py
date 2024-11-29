__all__ = ["Encoder"]

import typing

from .base import EmbeddingCache, Reranker

class Encoder(Reranker):

    def __init__(
        self,
        attr: typing.Union[str, typing.List[str]],
        key: str,
        encoder,
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            key=key,
            attr=attr,
            encoder=encoder,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        if k is None:
            k = self.k

        if k is None:
            k = len(self)

        if not documents and isinstance(q, str):
            return []

        if not documents and isinstance(q, list):
            return [[]]

        rank = self.encode_rank(
            embeddings_queries=self.encoder([q] if isinstance(q, str) else q),
            documents=[documents] if isinstance(q, str) else documents,
            k=k,
            batch_size=batch_size if batch_size is not None else self.batch_size,
        )

        return rank[0] if isinstance(q, str) else rank