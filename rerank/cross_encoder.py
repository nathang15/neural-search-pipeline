import typing
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import Reranker

class CrossEncoder(Reranker):
    def __init__(
        self,
        attr: typing.Union[str, typing.List[str]],
        key: str,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            key=key,
            attr=attr,
            encoder=None,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def score_pairs(self, queries: List[str], documents: List[str]) -> np.ndarray:
        """Score query-document pairs using cross-attention"""
        features = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        if torch.cuda.is_available():
            features = {k: v.cuda() for k, v in features.items()}
            
        with torch.no_grad():
            scores = self.model(**features).logits
            scores = torch.sigmoid(scores).cpu().numpy()
        
        return scores

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        if not documents:
            return [] if isinstance(q, str) else [[]]

        k = k or self.k or len(self)
        queries = [q] if isinstance(q, str) else q
        document_batches = [documents] if isinstance(q, str) else documents

        ranked_results = []
        for query, doc_batch in zip(queries, document_batches):
            if not doc_batch:
                ranked_results.append([])
                continue

            # Prepare text pairs for scoring
            doc_texts = [
                " ".join(str(doc.get(field, "")) for field in self.attr)
                for doc in doc_batch
            ]
            query_texts = [query] * len(doc_texts)
            
            # Get relevance scores
            scores = self.score_pairs(query_texts, doc_texts).flatten()
            
            # Sort and return top k results
            ranked_indices = np.argsort(-scores)[:k]
            
            batch_results = []
            for idx in ranked_indices:
                doc = doc_batch[idx].copy()
                doc["similarity"] = float(scores[idx])
                batch_results.append(doc)
            
            ranked_results.append(batch_results)

        return ranked_results[0] if isinstance(q, str) else ranked_results