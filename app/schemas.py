from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    source: str = Field(..., description="Unique name for this document, e.g. 'aha_guidelines_2024'.")
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    source: str | None = None


class SearchHit(BaseModel):
    score: float
    source: str | None
    chunk_index: int | None
    text: str | None


class ECGUploadRequest(BaseModel):
    signal: list[float] = Field(..., min_length=256, description="Single-lead ECG samples (mV).")
    fs: float = Field(..., gt=0, description="Sampling rate in Hz.")
    name: str | None = None
    r_peaks: list[int] | None = Field(None, description="Optional R-peak sample indices; auto-detected if omitted.")


class ECGUploadResponse(BaseModel):
    ecg_id: str
    duration_s: float


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    ecg_id: str | None = None


class ChatResponse(BaseModel):
    answer: str | None
    tool_calls: list[dict]
    sources: list[dict]
