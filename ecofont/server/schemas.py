from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ScriptName = Literal["hangul", "cherokee"]
DeviceName = Literal["auto", "cpu", "cuda"]
FontGenerationMethod = Literal["eco_research_guided", "ryman", "eco_diff"]
CodepointSetName = Literal["cherokee_full", "uploaded_cherokee", "hangul_full", "hangul_subset"]
ReturnFormat = Literal["zip"]
OcrEvalMode = Literal["none", "sample", "full"]


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=128, description="Characters to generate.")
    script: ScriptName | None = Field(None, description="Script hint. Inferred when omitted.")
    profile: str | None = Field(None, description="Generation profile name.")
    target_saving: float = Field(0.42, ge=0.0, le=0.8)
    diffusion_candidates: int = Field(2, ge=0, le=8)
    sample_steps: int | None = Field(48, ge=1, le=250)
    device: DeviceName = "auto"
    seed: int | None = Field(None, ge=0)
    image_size: int | None = Field(None, ge=32, le=512)
    font_size: int | None = Field(None, ge=16, le=512)
    use_vgg_style: bool = True
    include_candidate_preview: bool = False
    save_candidates_limit: int = Field(999, ge=0, le=5000)
    max_chars: int | None = Field(None, ge=1, le=128)
    font_path: str | None = Field(None, description="Optional server-local TTF/OTF path override.")


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    request: dict[str, Any]
    result_url: str | None = None


class ProfileInfo(BaseModel):
    name: str
    script: ScriptName
    config: str
    description: str
    default: bool = False


class FontGenerationEvaluation(BaseModel):
    ocr: bool = True
    ink: bool = True
    style: bool = False
    ocr_eval_mode: OcrEvalMode = "sample"
    eval_sample_size: int = Field(32, ge=1, le=512)
    ocr_lang: str | None = None
    ocr_psm: int | list[int] = Field(default_factory=lambda: [8, 6, 10])


class FontGenerationSpec(BaseModel):
    script: ScriptName
    method: FontGenerationMethod = "eco_research_guided"
    candidate_count: int = Field(20, ge=1, le=20)
    codepoint_set: CodepointSetName = "cherokee_full"
    return_format: ReturnFormat = "zip"
    target_saving: float = Field(0.42, ge=0.0, le=0.8)
    image_size: int = Field(96, ge=64, le=256)
    font_size: int = Field(76, ge=32, le=220)
    preview_text: str | None = Field(None, max_length=256)
    use_diffusion: bool = False
    evaluation: FontGenerationEvaluation = Field(default_factory=FontGenerationEvaluation)


class FontGenerationJobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    result_url: str


class FontGenerationJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    request: dict[str, Any]
    result_url: str | None = None


class FontGenerationMethodInfo(BaseModel):
    name: str
    supported_scripts: list[ScriptName]
    supported_codepoint_sets: list[CodepointSetName]
    max_candidate_count: int
    output_formats: list[ReturnFormat]
    description: str
