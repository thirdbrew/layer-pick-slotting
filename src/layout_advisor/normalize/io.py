"""Site profile YAML serialization."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml

from layout_advisor.models.site_profile import SiteProfile


def _encode(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def write_site_profile(profile: SiteProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encode(profile.model_dump(mode="json"))
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8")


_PATH_FIELDS = (
    "orders_path",
    "orders_slotting_path",
    "skus_path",
    "picks_path",
    "exclusions_path",
)


def _is_absolute_any_os(value: str) -> bool:
    # Path.is_absolute() is host-OS-specific, so a Windows-baked "C:\..." reads as
    # relative on POSIX (and vice versa) and would be mis-joined. Test both.
    import ntpath
    import posixpath

    return ntpath.isabs(value) or posixpath.isabs(value)


def read_site_profile(path: Path) -> SiteProfile:
    profile = SiteProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    # Resolve data-file paths against the profile's own directory so a moved or
    # freshly-cloned profile stays runnable. Absolute paths (legacy profiles,
    # from either OS) pass through unchanged.
    base = path.resolve().parent
    for field in _PATH_FIELDS:
        value = getattr(profile, field)
        if value and not _is_absolute_any_os(value):
            setattr(profile, field, str((base / value).resolve()))
    return profile
