from typing import List, Literal

from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    """
    The `EmbeddingResponse` class represents a response structure that likely contains information related to an embedding.

    Attributes:
    - `index`: An integer index that can be used to identify or order the embedding within a sequence of responses or a collection.
               For example, it might represent the position of the embedding in a list of multiple embeddings.
    - `embedding`: A list of floating-point numbers that constitutes the actual embedding vector. This vector typically
                   encodes some semantic or feature representation of an input, such as a word, sentence, or other data.
    - `object`: A literal value set to "embedding" which is used to identify the type of the object. This can be useful
                for serialization/deserialization processes or for distinguishing different types of responses in an API context.

    This class is likely used to standardize the format of responses when dealing with embeddings, making it easier to
    work with and validate the data received or sent related to embedding operations.
    """
    index: int
    embedding: List[float]
    object: Literal["embedding"] = "embedding"