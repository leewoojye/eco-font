from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import ProfileInfo, ScriptName


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    script: ScriptName
    config: str
    description: str
    default: bool = False

    def to_info(self) -> ProfileInfo:
        return ProfileInfo(
            name=self.name,
            script=self.script,
            config=self.config,
            description=self.description,
            default=self.default,
        )


PROFILES: dict[str, GenerationProfile] = {
    "hangul-jua": GenerationProfile(
        name="hangul-jua",
        script="hangul",
        config="configs/guided_jua.yaml",
        description="Hangul guided generation using the Jua source font.",
        default=True,
    ),
    "hangul-gothic": GenerationProfile(
        name="hangul-gothic",
        script="hangul",
        config="configs/guided.yaml",
        description="Hangul guided generation using Nanum Gothic.",
    ),
    "hangul-myeongjo": GenerationProfile(
        name="hangul-myeongjo",
        script="hangul",
        config="configs/guided_myeongjo.yaml",
        description="Hangul guided generation using Nanum Myeongjo.",
    ),
    "hangul-barunpen": GenerationProfile(
        name="hangul-barunpen",
        script="hangul",
        config="configs/guided_barunpen.yaml",
        description="Hangul guided generation using Nanum Barunpen.",
    ),
    "cherokee-noto": GenerationProfile(
        name="cherokee-noto",
        script="cherokee",
        config="configs/guided_cherokee.yaml",
        description="Cherokee guided generation using Noto Sans Cherokee.",
        default=True,
    ),
}


def profile_for_request(script: ScriptName, profile_name: str | None) -> GenerationProfile:
    if profile_name:
        try:
            profile = PROFILES[profile_name]
        except KeyError as exc:
            known = ", ".join(sorted(PROFILES))
            raise ValueError(f"unknown profile '{profile_name}'. Known profiles: {known}") from exc
        if profile.script != script:
            raise ValueError(f"profile '{profile.name}' is for {profile.script}, not {script}")
        return profile
    for profile in PROFILES.values():
        if profile.script == script and profile.default:
            return profile
    raise ValueError(f"no default profile configured for script '{script}'")


def profile_config_path(research_root: Path, profile: GenerationProfile) -> Path:
    return research_root / profile.config


def list_profile_infos() -> list[ProfileInfo]:
    return [profile.to_info() for profile in PROFILES.values()]

