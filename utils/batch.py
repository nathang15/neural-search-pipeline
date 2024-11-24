__all__ = ["iterate_with_progress", "iterate_in_batches"]

from typing import Union, List, Dict, Any, Generator
import numpy as np
import tqdm

def iterate_in_single(
    sequence: Union[
        Union[List[str], str],
        List[Dict[str, Any]],
    ],
    desc: str,
    tqdm_bar: bool = True,
) -> Generator:

    # Handle single string input case
    if isinstance(sequence, str):
        yield sequence
        return

    # Process sequence with or without progress bar
    if tqdm_bar:
        for element in tqdm.tqdm(
            sequence,
            position=0,
            desc=desc,
            total=len(sequence),
        ):
            yield element
    else:
        for element in sequence:
            yield element


def iterate_in_batches(
    sequence: Union[
        Union[
            Union[List[str], str],
            List[Dict[str, Any]],
        ],
        np.ndarray,
    ],
    batch_size: int,
    desc: str,
    tqdm_bar: bool = True,
) -> Generator:
    
    # Handle single string input case
    if isinstance(sequence, str):
        yield [sequence]
        return

    # Calculate batches using list comprehension
    batches = [
        sequence[start_idx : start_idx + batch_size]
        for start_idx in range(0, len(sequence), batch_size)
    ]
    
    # Calculate total number of batches for progress bar
    total_batches = 1 + len(sequence) // batch_size
    
    # Yield batches with or without progress bar
    if tqdm_bar:
        for batch in tqdm.tqdm(
            batches,
            position=0,
            desc=desc,
            total=total_batches,
        ):
            yield batch
    else:
        for batch in batches:
            yield batch