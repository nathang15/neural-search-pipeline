__all__ = ["Make"]

from .base import Make

class Pipeline(Make):
    def __init__(self, models: list) -> None:
        super().__init__(models=models)

    def __call__(self, q: str):
        query = {}
        for model in self.models:
            answer = model(q=q, **query)
            if isinstance(answer, list):
                query.update({"documents": answer})
            elif isinstance(answer, str):
                return answer
        return query["documents"]