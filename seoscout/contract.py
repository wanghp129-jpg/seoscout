"""Course contract validation for the unified ``seoscout run`` command."""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .collect import keyword_to_filename
from .generate import keyword_to_slug
from .core.config import Config


RISK_PATTERNS = [
    r"script",
    r"hack",
    r"exploit",
    r"pastebin",
    r"auto\s+farm",
    r"auto\s+quest",
    r"no\s+key",
    r"inf\s+money",
]


@dataclass(frozen=True)
class KeywordEntry:
    keyword: str
    category: str
    slug: str
    collected_path: Path
    english_path: Path


@dataclass(frozen=True)
class KeywordSpec:
    keywords_path: Path
    keywords_sha256: str
    topic_name: str
    languages: list[str]
    entries: list[KeywordEntry]


class ContractError(Exception):
    """Error with the required course exit code and pipeline stage."""

    def __init__(self, stage: str, exit_code: int, message: str):
        super().__init__(message)
        self.stage = stage
        self.exit_code = exit_code
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_output_dir(path: str | None) -> Path:
    if path is None or not str(path).strip():
        raise ContractError("input", 2, "--output-dir is required")
    output_dir = Path(path)
    if not output_dir.is_absolute():
        raise ContractError("input", 2, "--output-dir must be an absolute path")
    return output_dir


def load_keyword_spec(keywords_file: str, data_dir: str, out_dir: str) -> KeywordSpec:
    path = Path(keywords_file)
    if not path.is_absolute():
        raise ContractError("input", 2, "--keywords must be an absolute path")
    if not path.is_file():
        raise ContractError("input", 2, f"keywords file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError("input", 2, f"keywords JSON is invalid: {exc}") from exc

    if not isinstance(data, dict):
        raise ContractError("input", 2, "keywords JSON must be an object")

    topic = data.get("topic_name")
    if not isinstance(topic, str) or not topic.strip():
        raise ContractError("input", 2, "topic_name must be a non-empty string")
    topic = topic.strip()

    raw_entries: list[tuple[str, str]] = []
    if "categories" in data:
        categories = data.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ContractError("input", 2, "categories must be a non-empty array")
        for idx, category_obj in enumerate(categories):
            if not isinstance(category_obj, dict):
                raise ContractError("input", 2, f"categories[{idx}] must be an object")
            category = category_obj.get("category")
            if not isinstance(category, str) or not category.strip():
                raise ContractError("input", 2, f"categories[{idx}].category must be non-empty")
            keywords = category_obj.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                raise ContractError("input", 2, f"categories[{idx}].keywords must be non-empty")
            for keyword in keywords:
                raw_entries.append((keyword, category.strip()))
    elif "keywords" in data:
        keywords = data.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            raise ContractError("input", 2, "keywords must be a non-empty array")
        for keyword in keywords:
            raw_entries.append((keyword, ""))
    else:
        raise ContractError("input", 2, "keywords JSON must contain categories or keywords")

    if not raw_entries:
        raise ContractError("input", 2, "at least one keyword is required")

    languages = _validate_languages(data.get("languages", []))
    entries = _validate_entries(raw_entries, topic, Path(data_dir), Path(out_dir))

    return KeywordSpec(
        keywords_path=path,
        keywords_sha256=sha256_file(path),
        topic_name=topic,
        languages=languages,
        entries=entries,
    )


def _validate_languages(value: Any) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ContractError("input", 2, "languages must be an array when present")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ContractError("input", 2, "languages entries must be non-empty strings")
        normalized = item.strip().lower()
        if normalized in seen:
            raise ContractError("input", 2, f"duplicate language: {normalized}")
        seen.add(normalized)
        if normalized != "en":
            result.append(item.strip())
    return result


def _validate_entries(
    raw_entries: list[tuple[Any, str]],
    topic: str,
    data_dir: Path,
    out_dir: Path,
) -> list[KeywordEntry]:
    topic_lower = topic.lower()
    topic_prefix = f"{topic_lower} "
    seen_keywords: set[str] = set()
    slug_map: dict[str, str] = {}
    entries: list[KeywordEntry] = []

    for raw_keyword, category in raw_entries:
        if not isinstance(raw_keyword, str):
            raise ContractError("input", 2, "keyword must be a string")
        keyword = raw_keyword.strip()
        if not keyword:
            raise ContractError("input", 2, "keyword must be non-empty")
        if keyword != raw_keyword:
            raise ContractError("input", 2, f"keyword has leading or trailing whitespace: {keyword}")
        try:
            keyword.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ContractError("input", 2, f"keyword must be ASCII: {keyword}") from exc
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in keyword):
            raise ContractError("input", 2, f"keyword contains control characters: {keyword}")
        lower = keyword.lower()
        if lower in seen_keywords:
            raise ContractError("input", 2, f"duplicate keyword ignoring case: {keyword}")
        seen_keywords.add(lower)
        if lower == topic_lower:
            raise ContractError("input", 2, f"keyword cannot equal topic_name: {keyword}")
        if not lower.startswith(topic_prefix):
            raise ContractError("input", 2, f"keyword must start with topic_name: {keyword}")
        if re.search(r"(?<![a-z0-9])wiki(?![a-z0-9])", lower):
            raise ContractError("input", 2, f"wiki intent is not allowed: {keyword}")
        for pattern in RISK_PATTERNS:
            if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lower):
                raise ContractError("input", 2, f"risk intent is not allowed: {keyword}")

        slug = keyword_to_slug(keyword)
        if not slug:
            raise ContractError("input", 2, f"keyword maps to empty slug: {keyword}")
        if not re.fullmatch(r"[a-z0-9-]+", slug):
            raise ContractError("input", 2, f"keyword maps to unsafe slug: {keyword}")
        if slug in slug_map and slug_map[slug] != keyword:
            raise ContractError("input", 2, f"slug collision: {slug}")
        slug_map[slug] = keyword

        category_slug = category.lower().replace(" ", "-") if category else ""
        filename = keyword_to_filename(keyword)
        if category_slug:
            collected_path = Path(out_dir) / "collected" / category_slug / f"{filename}.json"
        else:
            collected_path = Path(out_dir) / "collected" / f"{filename}.json"
        try:
            english_path = Config.article_path("en", slug, category, data_dir)
        except ValueError as exc:
            raise ContractError("input", 2, str(exc)) from exc
        entries.append(KeywordEntry(keyword, category, slug, collected_path, english_path))

    return entries


def validate_collected_material(spec: KeywordSpec) -> None:
    missing = [entry for entry in spec.entries if not entry.collected_path.is_file()]
    if missing:
        raise ContractError("search_collect", 3, f"missing collected material for {len(missing)} keyword(s)")
    for entry in spec.entries:
        try:
            data = json.loads(entry.collected_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ContractError("search_collect", 3, f"invalid collected material: {entry.collected_path}") from exc
        if not isinstance(data, dict) or data.get("total_sources", 0) <= 0:
            raise ContractError("search_collect", 3, f"empty collected material: {entry.collected_path}")


def validate_search_results(spec: KeywordSpec, out_dir: str) -> None:
    path = Path(out_dir) / "search_results.json"
    if not path.is_file():
        raise ContractError("search_collect", 3, f"search results missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError("search_collect", 3, f"search results invalid: {path}") from exc
    rows = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ContractError("search_collect", 3, "search results keywords must be an array")
    by_keyword = {row.get("keyword"): row for row in rows if isinstance(row, dict)}
    for entry in spec.entries:
        row = by_keyword.get(entry.keyword)
        if not row:
            raise ContractError("search_collect", 3, f"search result missing for keyword: {entry.keyword}")
        youtube_items = row.get("youtube", {}).get("items", [])
        web_items = row.get("web", {}).get("items", [])
        if not youtube_items and not web_items:
            raise ContractError("search_collect", 3, f"search returned no usable sources: {entry.keyword}")


def validate_english_articles(spec: KeywordSpec) -> None:
    _validate_article_set(spec, "en", 4)


def validate_translated_articles(spec: KeywordSpec) -> None:
    for language in spec.languages:
        _validate_article_set(spec, language, 5)


def _validate_article_set(spec: KeywordSpec, language: str, exit_code: int) -> None:
    expected = {entry.slug for entry in spec.entries}
    base = _articles_root(spec.entries[0].english_path) if spec.entries else None
    if base is None:
        raise ContractError("english" if language == "en" else "translation", exit_code, "no expected articles")
    lang_dir = base / language
    actual_paths = list(lang_dir.glob("**/*.mdx")) if lang_dir.exists() else []
    actual = [p.stem for p in actual_paths]
    actual_set = set(actual)
    if len(actual) != len(actual_set):
        raise ContractError(_stage_for_language(language), exit_code, f"duplicate slug in {language} articles")
    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    if missing or extra:
        raise ContractError(
            _stage_for_language(language),
            exit_code,
            f"{language} slug set mismatch; missing={len(missing)} extra={len(extra)}",
        )
    for entry in spec.entries:
        if language == "en":
            path = entry.english_path
        else:
            path = _translation_path(entry, language)
        validate_mdx_file(path, entry.slug, exit_code, _stage_for_language(language), entry.category or None)


def _stage_for_language(language: str) -> str:
    return "english" if language == "en" else "translation"


def _translation_path(entry: KeywordEntry, language: str) -> Path:
    en_dir = _articles_root(entry.english_path)
    rel = entry.english_path.relative_to(en_dir / "en")
    return en_dir / language / rel


def _articles_root(path: Path) -> Path:
    parts = path.parts
    if "articles" not in parts:
        raise ContractError("english", 4, f"article path is outside articles tree: {path}")
    idx = len(parts) - 1 - list(reversed(parts)).index("articles")
    return Path(*parts[: idx + 1])


def validate_mdx_file(path: Path, expected_slug: str, exit_code: int = 4, stage: str = "english", expected_category: str | None = None) -> None:
    if not path.is_file():
        raise ContractError(stage, exit_code, f"missing MDX file: {path}")
    if path.stem != expected_slug:
        raise ContractError(stage, exit_code, f"MDX filename does not match slug: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ContractError(stage, exit_code, f"empty MDX file: {path}")
    if "TODO" in content or "{{" in content or "}}" in content:
        raise ContractError(stage, exit_code, f"unreplaced placeholder detected: {path}")
    body = _validate_js_metadata(content, path, stage, exit_code, expected_category)
    if not body.strip():
        raise ContractError(stage, exit_code, f"empty MDX body: {path}")
    _validate_mdx_components(body, path, stage, exit_code)


def _validate_js_metadata(content: str, path: Path, stage: str, exit_code: int, expected_category: str | None = None) -> str:
    if not content.lstrip().startswith("export const metadata = {"):
        raise ContractError(stage, exit_code, f"missing JS metadata export: {path}")
    lines = content.splitlines()
    if not lines:
        raise ContractError(stage, exit_code, f"missing metadata lines: {path}")
    end_idx = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() in ("}", "};"):
            end_idx = idx
            break
    if end_idx is None:
        raise ContractError(stage, exit_code, f"metadata block is not closed: {path}")
    metadata_lines = lines[1:end_idx]
    fields = {}
    for line in metadata_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = re.match(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*(.+?)(,)?$", stripped)
        if not match:
            raise ContractError(stage, exit_code, f"invalid metadata line: {path}")
        key = match.group(1)
        value = match.group(2).strip()
        if value.startswith("'"):
            raise ContractError(stage, exit_code, f"single-quoted metadata string is not allowed: {path}")
        if value.startswith('"') and not _double_quoted_value_closed(value):
            raise ContractError(stage, exit_code, f"unterminated double-quoted metadata string: {path}")
        if key in {"title", "description", "category", "date", "slug", "translationKey"}:
            if key in fields:
                raise ContractError(stage, exit_code, f"duplicate metadata field {key}: {path}")
            try:
                parsed = json.loads(value)
            except ValueError as exc:
                raise ContractError(stage, exit_code, f"invalid metadata string {key}: {path}") from exc
            if not isinstance(parsed, str) or not parsed.strip():
                raise ContractError(stage, exit_code, f"empty metadata string {key}: {path}")
            fields[key] = parsed
    for key in ("title", "description", "category", "date"):
        if key not in fields:
            raise ContractError(stage, exit_code, f"missing metadata field {key}: {path}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["date"]):
        raise ContractError(stage, exit_code, f"invalid metadata date: {path}")
    try:
        datetime.strptime(fields["date"], "%Y-%m-%d")
    except ValueError as exc:
        raise ContractError(stage, exit_code, f"invalid metadata date: {path}") from exc
    if expected_category and fields["category"] != expected_category.lower().replace(" ", "-"):
        raise ContractError(stage, exit_code, f"metadata category mismatch: {path}")
    for key in ("slug", "translationKey"):
        if key in fields and fields[key] != path.stem:
            raise ContractError(stage, exit_code, f"metadata {key} mismatch: {path}")
    return "\n".join(lines[end_idx + 1 :])


def _double_quoted_value_closed(value: str) -> bool:
    escaped = False
    for idx, ch in enumerate(value[1:], start=1):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            return True
    return False


def _validate_mdx_components(body: str, path: Path, stage: str, exit_code: int) -> None:
    for match in re.finditer(r"</?([A-Z][A-Za-z0-9_.]*)\b", body):
        name = match.group(1)
        if not re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:\.[A-Z][A-Za-z0-9]*)?", name):
            raise ContractError(stage, exit_code, f"invalid MDX component name {name}: {path}")


def build_traceability_manifest(spec: KeywordSpec, out_dir: str) -> dict[str, Any]:
    mappings = []
    for entry in spec.entries:
        translations: dict[str, dict[str, str]] = {}
        for language in spec.languages:
            translated = _translation_path(entry, language)
            _require_file(translated, "manifest")
            translations[language] = {
                "mdx_path": str(translated.resolve()),
                "mdx_sha256": sha256_file(translated),
            }

        _require_file(entry.collected_path, "manifest")
        _require_file(entry.english_path, "manifest")
        mappings.append({
            "keyword": entry.keyword,
            "category": entry.category,
            "slug": entry.slug,
            "collected_material_path": str(entry.collected_path.resolve()),
            "collected_material_sha256": sha256_file(entry.collected_path),
            "english_mdx_path": str(entry.english_path.resolve()),
            "english_mdx_sha256": sha256_file(entry.english_path),
            "translations": translations,
        })

    return {
        "keywords_path": str(spec.keywords_path.resolve()),
        "keywords_sha256": spec.keywords_sha256,
        "topic_name": spec.topic_name,
        "keyword_count": len(spec.entries),
        "languages": spec.languages,
        "mappings": mappings,
    }


def write_traceability_manifest(spec: KeywordSpec, out_dir: str) -> Path:
    manifest = build_traceability_manifest(spec, out_dir)
    path = Path(out_dir) / "traceability_manifest.json"
    _atomic_json(path, manifest)
    validate_traceability_manifest(path)
    return path


def validate_traceability_manifest(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError("manifest", 6, f"traceability manifest is invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ContractError("manifest", 6, "traceability manifest must be an object")
    mappings = data.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ContractError("manifest", 6, "traceability manifest mappings must be non-empty")
    for mapping in mappings:
        for key in ("collected_material_path", "english_mdx_path"):
            p = Path(mapping.get(key, ""))
            if not p.is_absolute() or not p.is_file():
                raise ContractError("manifest", 6, f"manifest references missing file: {key}")
        if sha256_file(Path(mapping["collected_material_path"])) != mapping.get("collected_material_sha256"):
            raise ContractError("manifest", 6, "collected material SHA mismatch")
        if sha256_file(Path(mapping["english_mdx_path"])) != mapping.get("english_mdx_sha256"):
            raise ContractError("manifest", 6, "English MDX SHA mismatch")
        translations = mapping.get("translations", {})
        if not isinstance(translations, dict):
            raise ContractError("manifest", 6, "translations must be an object")
        for item in translations.values():
            p = Path(item.get("mdx_path", ""))
            if not p.is_absolute() or not p.is_file():
                raise ContractError("manifest", 6, "manifest references missing translated MDX")
            if sha256_file(p) != item.get("mdx_sha256"):
                raise ContractError("manifest", 6, "translated MDX SHA mismatch")


def _require_file(path: Path, stage: str) -> None:
    if not path.is_file():
        raise ContractError(stage, 6 if stage == "manifest" else 4, f"required file missing: {path}")


def write_run_status(
    out_dir: str,
    *,
    status: str,
    spec: KeywordSpec | None,
    output_root: str,
    started_at: str,
    failure: dict[str, Any] | None = None,
    stage_counts: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "status": status,
        "topic_name": spec.topic_name if spec else "",
        "keywords_path": str(spec.keywords_path.resolve()) if spec else "",
        "keywords_sha256": spec.keywords_sha256 if spec else "",
        "output_root": output_root,
        "keyword_count": len(spec.entries) if spec else 0,
        "languages": spec.languages if spec else [],
        "started_at": started_at,
        "completed_at": utc_now() if status == "completed" else None,
        "stage_counts": stage_counts or {},
    }
    if failure is not None:
        payload["failure"] = failure
    path = Path(out_dir) / "run_status.json"
    _atomic_json(path, payload)
    return path


def validate_run_status(path: Path, expect_success: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if expect_success:
        if data.get("status") != "completed":
            raise ContractError("manifest", 6, "run_status success status missing")
        if "failure" in data:
            raise ContractError("manifest", 6, "successful run_status must not require failure")
    else:
        failure = data.get("failure")
        if data.get("status") != "failed" or not isinstance(failure, dict):
            raise ContractError("manifest", 6, "failed run_status failure object missing")
        for key in ("stage", "exit_code", "message"):
            if key not in failure:
                raise ContractError("manifest", 6, f"failed run_status missing failure.{key}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
