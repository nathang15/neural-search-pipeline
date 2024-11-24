__all__ = ["TopK"]

import typing


class TopK:
    """Filter top k documents in pipeline"""

    def __init__(self, k: int):
        self.k = k

    def __repr__(self) -> str:
        repr = f"Filter {self.__class__.__name__}"
        repr += f"\n\tk: {self.k}"
        return repr

    def __call__(
        self,
        documents: typing.Union[typing.List[typing.List[typing.Dict[str, str]]]],
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        if not documents:
            return []

        if isinstance(documents[0], list):
            return [document[: self.k] for document in documents]

        return documents[: self.k]