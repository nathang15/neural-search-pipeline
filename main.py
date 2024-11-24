from pprint import pprint as print
import utils, retrieve
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Any
import torch

def main(documents: List[Dict[str, Any]], query: str = "", quantize: bool = False) -> Dict:
    metrics = {}
    
    # Time model loading
    start_time = time.perf_counter()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    metrics['model_load'] = time.perf_counter() - start_time
    
    # Time quantization
    if quantize:
        start_time = time.perf_counter()
        encoder = utils.quantizer(model)
        metrics['quantization'] = time.perf_counter() - start_time
    else:
        encoder = model
        metrics['quantization'] = 0.0
        
    # Time retriever initialization
    start_time = time.perf_counter()
    retriever = retrieve.Encoder(
        encoder=encoder.encode,
        key="id",
        attr=["title", "article"],
    )
    metrics['retriever_init'] = time.perf_counter() - start_time
    
    # Time document indexing
    start_time = time.perf_counter()
    retriever = retriever.add(documents)
    metrics['indexing'] = time.perf_counter() - start_time
    
    # Time search query
    start_time = time.perf_counter()
    results = retriever(query)
    metrics['search'] = time.perf_counter() - start_time
    
    # Calculate total time
    metrics['total'] = sum(metrics.values())
    
    return {
        'metrics': metrics,
        'results': results
    }

if __name__ == "__main__":
    documents = [
        {
            "id": 0,
            "article": "Ho Chi Minh City, better known as Saigon, is the most populous city in Vietnam, with a population of around 10 million in 2023.",
            "title": "Ho Chi Minh City",
            "url": "https://en.wikipedia.org/wiki/Ho_Chi_Minh_City"
        },
        {
            "id": 1,
            "article": "The city's geography is defined by rivers and canals, of which the largest is Saigon River.",
            "title": "Ho Chi Minh City",
            "url": "https://en.wikipedia.org/wiki/Ho_Chi_Minh_City"
        },
        {
            "id": 2,
            "article": "As a municipality, Ho Chi Minh City consists of 16 urban districts, five rural districts, and one municipal city (sub-city). As the largest financial centre in Vietnam, Ho Chi Minh City has the highest gross regional domestic product out of all Vietnam provinces and municipalities,[8] contributing around a quarter of the country's total GDP.[9] Ho Chi Minh City's metropolitan area is ASEAN's 6th largest economy, also the biggest outside an ASEAN country capital.",
            "title": "Ho Chi Minh City",
            "url": "https://en.wikipedia.org/wiki/Ho_Chi_Minh_City"
        }
    ]
    
    # Run with timing
    results = main(documents, "Ho Chi Minh has 16 urban districts and five rural districts.", quantize=False)
    
    # Print detailed timing report
    print("\nPerformance Metrics:")
    print("=" * 50)
    print(f"Model Loading:      {results['metrics']['model_load']:.4f}s")
    print(f"Quantization:       {results['metrics']['quantization']:.4f}s")
    print(f"Retriever Setup:    {results['metrics']['retriever_init']:.4f}s")
    print(f"Document Indexing:  {results['metrics']['indexing']:.4f}s")
    print(f"Search Query:       {results['metrics']['search']:.4f}s")
    print(f"Total Time:         {results['metrics']['total']:.4f}s")

    print("\nSearch Results:")
    print("=" * 50)
    print(results['results'])