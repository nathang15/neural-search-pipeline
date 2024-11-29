import typing

from .base import Reranker

__all__ = ["DPR"]

class DPR(Reranker):
    def __init__(
        self,
        attr: typing.Union[str, typing.List[str]],
        key: str,
        encoder,
        query_encoder,
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
        self.query_encoder = query_encoder

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, typing.Any]]],
        typing.List[typing.Dict[str, typing.Any]],
    ]:

        k = k or self.k or len(self)
        
        if not documents:
            return [[]] if isinstance(q, list) else []

        queries = [q] if isinstance(q, str) else q
        
        embeddings_queries = self.query_encoder(queries)
        
        processed_documents = (
            [documents] if isinstance(q, str) else documents
        )
        
        rank = self.encode_rank(
            embeddings_queries=embeddings_queries,
            documents=processed_documents,
            k=k,
            batch_size=batch_size or self.batch_size,
        )

        return rank[0] if isinstance(q, str) else rank