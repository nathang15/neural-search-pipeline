import typing
import numpy as np
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

    def _encoder(self, documents: typing.List[typing.Dict[str, str]]) -> np.ndarray:
        """Properly formats and encodes document content."""
        text_inputs = []
        for doc in documents:
            field_texts = []
            for field in self.attr:
                field_value = doc.get(field, "")
                if isinstance(field_value, list):
                    field_value = " ".join(str(item) for item in field_value)
                elif not isinstance(field_value, str):
                    field_value = str(field_value)
                field_texts.append(field_value)
            text_inputs.append(" ".join(field_texts))
        return self.encoder(text_inputs)

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
        if not documents:
            return [] if isinstance(q, str) else [[]]

        k = k or self.k or len(self)
        queries = [q] if isinstance(q, str) else q
        processed_documents = [documents] if isinstance(q, str) else documents

        try:
            # Process queries
            embeddings_queries = self.query_encoder([
                query if isinstance(query, str) else str(query) 
                for query in queries
            ])
            
            rank = self.encode_rank(
                embeddings_queries=embeddings_queries,
                documents=processed_documents,
                k=k,
                batch_size=batch_size or self.batch_size,
            )
            
            return rank[0] if isinstance(q, str) else rank
            
        except Exception as e:
            print(f"DPR encoding error: {e}")
            return [] if isinstance(q, str) else [[]]