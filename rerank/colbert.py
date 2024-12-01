import typing
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base import Reranker

class ColBERT(Reranker):
    def __init__(
        self,
        attr: typing.Union[str, typing.List[str]],
        key: str,
        model_name: str = "bert-base-uncased",
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 32
    ) -> None:
        super().__init__(
            key=key,
            attr=attr,
            encoder=None,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_text(self, texts: typing.List[str], max_length: int = 512) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                
        return embeddings

    def compute_similarity(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        similarity = torch.matmul(query_emb, doc_emb.transpose(-2, -1))
        return similarity.max(dim=-1)[0].mean(dim=-1)

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

            query_emb = self.encode_text([query], max_length=256)
            
            doc_texts = [
                " ".join(str(doc.get(field, "")) for field in self.attr)
                for doc in doc_batch
            ]
            
            scores = []
            for i in range(0, len(doc_texts), self.batch_size):
                batch = doc_texts[i:i + self.batch_size]
                doc_emb = self.encode_text(batch)
                batch_scores = self.compute_similarity(query_emb, doc_emb)
                scores.extend(batch_scores.cpu().numpy())
            
            # Rank and return top k results
            scores = np.array(scores)
            ranked_indices = np.argsort(-scores)[:k]
            
            batch_results = []
            for idx in ranked_indices:
                doc = doc_batch[idx].copy()
                doc["similarity"] = float(scores[idx])
                batch_results.append(doc)
            
            ranked_results.append(batch_results)

        return ranked_results[0] if isinstance(q, str) else ranked_results