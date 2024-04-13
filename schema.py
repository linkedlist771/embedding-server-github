from pydantic import BaseModel, Field
from typing import List, Optional, Union, TypeVar
from enum import Enum

"""
The schema here totally follows the OpenAI API response format.
"""


class CollectionType(str, Enum):
    history = "history"
    knowledge = "knowledge"


class BaseRequest(BaseModel):
    model: Optional[str] = Field(None, example="all-mpnet-base-v2")


class EmbeddingRequest(BaseRequest):
    # ... means this parameter is required
    input: Union[List[str], str] = Field(..., example=["Your text string goes here"])


class BaseResponse(BaseModel):
    object: str = Field(..., example="list")


class Usage(BaseModel):
    """
    Don't use the tokenizer, so just the prompt's length.
    """

    prompt_tokens: int = Field(..., example=15)
    total_tokens: int = Field(..., example=15)


class Embedding(BaseModel):
    object: str = Field(default="embedding", example="embedding")
    index: int = Field(..., example=0)
    embedding: List[float] = Field(..., example=[-0.007021796])


class EmbeddingResponse(BaseResponse):
    data: List[Embedding] = Field(...)
    model: str = Field(default="all-mpnet-base-v2", example="all-mpnet-base-v2")
    usage: Usage = Field(...)


T = TypeVar("T")


class DocumentBaseResponse(BaseModel):
    message: str = Field(..., example="Document added successfully.")
    data: Optional[T] = Field(None, example=None)
    code: int = Field(..., example=200)


class DocumentAddingRequest(BaseRequest):
    emb_model_type: str = Field(..., example="default")
    custom_index: Optional[int] = Field(None, example=0)
    collection_type: CollectionType = Field(..., example=CollectionType.history)
    documents: List[str] = Field(..., example=["document1", "document2"])
    metadatas: List[dict] = Field(..., example=[{"key": "value"}, {"key": "value"}])
    ids: List[str] = Field(..., example=["id1", "id2"])


class DocumentAddingResponse(DocumentBaseResponse):
    pass


class DocumentQueryRequest(BaseRequest):
    emb_model_type: str = Field(..., example="default")
    custom_index: Optional[int] = Field(None, example=0)
    collection_type: CollectionType = Field(..., example=CollectionType.history)
    query: str = Field(..., example="query string")
    n_results: int = Field(..., example=10)
    meta_filter: Optional[dict] = Field(
        None, example={"$and": [{"agent": "Dobby"}, {"type": "quote"}]}
    )


class DocumentQueryResponse(DocumentBaseResponse):
    pass


class DocumentOverviewResponse(DocumentBaseResponse):
    pass
