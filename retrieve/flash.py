import typing
from collections import Counter
from itertools import chain

from flashtext import KeywordProcessor

from ..utils import iterate_in_single
from .base import Retriever

__all__ = ["Flash"]
class Flash(Retriever):

    def __init__(
        self,
        key: str,
        attr: typing.Union[str, list],
        keywords: KeywordProcessor = None,
        lowercase: bool = True,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key=key, attr=attr, k=k, batch_size=1)
        self.documents = {}
        self.keywords = keywords or KeywordProcessor()
        self.lowercase = lowercase
        self._keyword_cache = {}

    def add(self, documents: typing.List[typing.Dict[str, str]], **kwargs) -> "Flash":
        for document in documents:
            for field in self.attr:
                if field not in document:
                    continue

                content = document[field]
                if isinstance(content, str):
                    content = [content]

                content = [
                    word.lower() if self.lowercase else word 
                    for word in (content if isinstance(content, list) else [content])
                ]

                for word in content:
                    if word not in self.documents:
                        self.documents[word] = []
                    self.documents[word].append({self.key: document[self.key]})
                    self.keywords.add_keyword(word)

        return self

    def _extract_keywords(self, query: str) -> list:
        if query not in self._keyword_cache:
            processed_query = query.lower() if self.lowercase else query
            self._keyword_cache[query] = self.keywords.extract_keywords(processed_query)
        return self._keyword_cache[query]

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int] = None,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> list:
        rank = []

        for batch in iterate_in_single(
            q, desc=f"{self.__class__.__name__} retriever", tqdm_bar=tqdm_bar
        ):
            keywords = self._extract_keywords(batch)
            
            match = list(
                chain.from_iterable(
                    self.documents.get(tag, []) for tag in keywords
                )
            )

            scores = Counter(doc[self.key] for doc in match)
            total = len(match)

            documents = [
                {self.key: key, "Similarity": count / total}
                for key, count in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]

            documents = documents[:k] if k is not None else documents
            rank.append(documents)

        return rank[0] if isinstance(q, str) else rank