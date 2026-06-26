from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from .profiles import GenerationProfile, profile_config_path, profile_for_request
from .schemas import GenerateRequest, ScriptName


ECO_FONT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ECO_FONT_ROOT / "eco_research_hangul"
RESEARCH_SRC = RESEARCH_ROOT / "src"
API_CONFIG_ROOT = RESEARCH_ROOT / "configs"
API_OUTPUT_ROOT = RESEARCH_ROOT / "outputs" / "api_jobs"


def _ensure_research_import_path() -> None:
    path = str(RESEARCH_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)


def _normalized_chars(text: str, max_chars: int | None = None) -> str:
    chars: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch.isspace():
            continue
        if ch in seen:
            continue
        seen.add(ch)
        chars.append(ch)
        if max_chars is not None and len(chars) >= max_chars:
            break
    if not chars:
        raise ValueError("text must contain at least one non-space character")
    return "".join(chars)


def _is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return (
        0xAC00 <= cp <= 0xD7A3
        or 0x1100 <= cp <= 0x11FF
        or 0x3130 <= cp <= 0x318F
        or 0xA960 <= cp <= 0xA97F
        or 0xD7B0 <= cp <= 0xD7FF
    )


def _is_cherokee(ch: str) -> bool:
    cp = ord(ch)
    return 0x13A0 <= cp <= 0x13FF or 0xAB70 <= cp <= 0xABBF


def infer_script(text: str) -> ScriptName:
    chars = [ch for ch in text if not ch.isspace()]
    if chars and all(_is_hangul(ch) for ch in chars):
        return "hangul"
    if chars and all(_is_cherokee(ch) for ch in chars):
        return "cherokee"
    raise ValueError("script could not be inferred. Send only Hangul or only Cherokee characters, or set script explicitly.")


def validate_script(text: str, script: ScriptName) -> None:
    chars = [ch for ch in text if not ch.isspace()]
    if script == "hangul" and not all(_is_hangul(ch) for ch in chars):
        raise ValueError("hangul requests must contain only Hangul characters")
    if script == "cherokee" and not all(_is_cherokee(ch) for ch in chars):
        raise ValueError("cherokee requests must contain only Cherokee characters")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _asset_url(job_id: str, rel_path: str | Path) -> str:
    rel = Path(rel_path).as_posix().lstrip("/")
    return f"/v1/assets/{quote(job_id)}/{quote(rel, safe='/')}"


def _resolve_research_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else RESEARCH_ROOT / path)


class EcoResearchRunner:
    def __init__(self, eco_font_root: Path = ECO_FONT_ROOT) -> None:
        self.eco_font_root = eco_font_root
        self.research_root = eco_font_root / "eco_research_hangul"
        self.api_config_root = self.research_root / "configs"
        self.api_output_root = self.research_root / "outputs" / "api_jobs"

    def prepare_request(self, request: GenerateRequest) -> tuple[ScriptName, GenerationProfile, str]:
        chars = _normalized_chars(request.text, request.max_chars)
        script = request.script or infer_script(chars)
        validate_script(chars, script)
        profile = profile_for_request(script, request.profile)
        return script, profile, chars

    def build_config(self, job_id: str, request: GenerateRequest) -> tuple[Path, Path, dict[str, Any]]:
        script, profile, chars = self.prepare_request(request)
        source_config = profile_config_path(self.research_root, profile)
        if not source_config.exists():
            raise FileNotFoundError(f"profile config not found: {source_config}")
        cfg = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
        data_cfg = dict(cfg.get("data", {}))
        inf_cfg = dict(cfg.get("guided_inference", {}))

        if request.image_size is not None:
            data_cfg["image_size"] = int(request.image_size)
        if request.font_size is not None:
            data_cfg["font_size"] = int(request.font_size)

        if request.seed is not None:
            cfg["seed"] = int(request.seed)
        inf_cfg["chars"] = chars
        inf_cfg.pop("charset_file", None)
        inf_cfg["output_dir"] = f"outputs/api_jobs/{job_id}/guided"
        inf_cfg["target_saving"] = float(request.target_saving)
        inf_cfg["diffusion_candidates"] = int(request.diffusion_candidates)
        inf_cfg["device"] = request.device
        inf_cfg["use_vgg_style"] = bool(request.use_vgg_style)
        inf_cfg["save_top_candidates"] = int(request.save_candidates_limit)
        if request.sample_steps is not None:
            inf_cfg["sample_steps"] = int(request.sample_steps)
        if request.font_path:
            inf_cfg["font"] = request.font_path

        inf_cfg["font"] = _resolve_research_path(str(inf_cfg["font"]))
        if inf_cfg.get("preview_label_font"):
            inf_cfg["preview_label_font"] = _resolve_research_path(str(inf_cfg["preview_label_font"]))

        cfg["data"] = data_cfg
        cfg["guided_inference"] = inf_cfg
        cfg.setdefault("api", {})
        cfg["api"].update(
            {
                "job_id": job_id,
                "script": script,
                "profile": profile.name,
                "request_text": request.text,
                "normalized_chars": chars,
            }
        )

        self.api_config_root.mkdir(parents=True, exist_ok=True)
        self.api_output_root.mkdir(parents=True, exist_ok=True)
        config_path = self.api_config_root / f"api_job_{job_id}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        job_root = self.api_output_root / job_id
        return config_path, job_root, {"script": script, "profile": profile.name, "chars": chars}

    def run(self, job_id: str, request: GenerateRequest) -> dict[str, Any]:
        _ensure_research_import_path()
        from eco_research_hangul.candidate_preview import generate_candidate_preview
        from eco_research_hangul.guided import run_guided_inference_from_config
        from eco_research_hangul.report import make_report

        config_path, job_root, prepared = self.build_config(job_id, request)
        manifest_path = run_guided_inference_from_config(config_path)
        guided_root = manifest_path.parent

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        label_font = cfg["guided_inference"].get("preview_label_font") or cfg["guided_inference"].get("font")
        summary = make_report(guided_root, guided_root / "contact_sheet.png", label_font=label_font)

        preview_manifest = None
        if request.include_candidate_preview:
            preview_manifest = generate_candidate_preview(
                config_path,
                output_dir=f"outputs/api_jobs/{job_id}/candidate_preview",
                chars=prepared["chars"],
                max_chars=request.max_chars,
            )

        result = self.collect_result(
            job_id=job_id,
            job_root=job_root,
            guided_root=guided_root,
            summary=summary,
            prepared=prepared,
            preview_manifest=preview_manifest,
        )
        result_path = job_root / "api_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def collect_result(
        self,
        job_id: str,
        job_root: Path,
        guided_root: Path,
        summary: dict[str, Any],
        prepared: dict[str, Any],
        preview_manifest: Path | None,
    ) -> dict[str, Any]:
        inference_rows = _read_jsonl(guided_root / "inference_manifest.jsonl")
        candidate_rows = _read_jsonl(guided_root / "candidate_manifest.jsonl")
        candidates_by_char: dict[str, list[dict[str, Any]]] = {}
        for row in candidate_rows:
            char_id = str(row.get("char_id"))
            item = {
                "name": row.get("name"),
                "metrics": row.get("metrics"),
                "image": _asset_url(job_id, Path("guided") / row["path"]) if row.get("path") else None,
            }
            candidates_by_char.setdefault(char_id, []).append(item)

        characters = []
        for row in inference_rows:
            char_id = row["char_id"]
            characters.append(
                {
                    "char": row["char"],
                    "char_id": char_id,
                    "source": _asset_url(job_id, Path("guided") / row["source"]),
                    "generated": _asset_url(job_id, Path("guided") / row["generated"]),
                    "metrics": row.get("metrics"),
                    "candidates": candidates_by_char.get(char_id, []),
                }
            )

        result: dict[str, Any] = {
            "job_id": job_id,
            "status": "completed",
            "script": prepared["script"],
            "profile": prepared["profile"],
            "text": prepared["chars"],
            "output_root": str(job_root),
            "summary": summary,
            "contact_sheet": _asset_url(job_id, "guided/contact_sheet.png"),
            "manifests": {
                "inference": _asset_url(job_id, "guided/inference_manifest.jsonl"),
                "candidates": _asset_url(job_id, "guided/candidate_manifest.jsonl"),
                "metrics_summary": _asset_url(job_id, "guided/metrics_summary.json"),
            },
            "characters": characters,
        }
        if preview_manifest is not None:
            result["candidate_preview"] = {
                "manifest": _asset_url(job_id, "candidate_preview/candidate_preview_manifest.jsonl"),
                "contact_sheet": _asset_url(job_id, "candidate_preview/contact_sheet.png"),
            }
        return result

    def resolve_asset(self, job_id: str, asset_path: str) -> Path:
        root = (self.api_output_root / job_id).resolve()
        path = (root / asset_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("asset path escapes the job output directory")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(asset_path)
        return path
