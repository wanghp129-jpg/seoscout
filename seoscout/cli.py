#!/usr/bin/env python3
"""
seoscout CLI — unified entry point.

Usage:
    seoscout search --keywords FILE
    seoscout collect --keywords FILE
    seoscout generate --keywords FILE [--prompt FILE] [--overwrite] [--test]
    seoscout translate --keywords FILE --lang es,pt,de [--prompt FILE] [--overwrite] [--test]
    seoscout run --keywords FILE --output-dir DIR
"""

import asyncio
import argparse
import json
import os
import sys
import fcntl
import hashlib
from pathlib import Path

from . import __version__


def _derive_project(keywords_file: str) -> str:
    """Derive project name from topic_name in JSON, or fall back to filename."""
    try:
        with open(keywords_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        topic = data.get('topic_name', '').strip()
        if topic:
            return topic.replace(' ', '_').lower()
    except Exception:
        pass
    base = os.path.splitext(os.path.basename(keywords_file))[0]
    return base.replace(' ', '_').lower()


def main():
    parser = argparse.ArgumentParser(
        prog="seoscout",
        description="Keyword research, content collection, article generation & translation CLI for SEO"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── search ──
    search_parser = subparsers.add_parser(
        "search",
        help="Search keywords on YouTube and Google → search_results.json"
    )
    search_parser.add_argument(
        "--keywords", "-k", required=True,
        help="Path to keywords JSON file"
    )

    # ── collect ──
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect YouTube transcripts and web content from search results"
    )
    collect_parser.add_argument(
        "--project", "-p",
        help="Project name (auto-derived from keywords file if not set)"
    )
    collect_parser.add_argument(
        "--keywords", "-k",
        help="Keywords JSON file (used to derive project name if --project not set)"
    )

    # ── generate ──
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate Markdown articles from collected material using LLM"
    )
    gen_parser.add_argument(
        "--keywords", "-k", required=True,
        help="Path to keywords JSON file"
    )
    gen_parser.add_argument(
        "--prompt",
        help="Path to custom prompt template (default: built-in)"
    )
    gen_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing articles"
    )
    gen_parser.add_argument(
        "--test", action="store_true",
        help="Test mode: only generate 2 articles"
    )

    # ── translate ──
    trans_parser = subparsers.add_parser(
        "translate",
        help="Translate English articles to other languages using LLM"
    )
    trans_parser.add_argument(
        "--keywords", "-k", required=True,
        help="Path to keywords JSON file (used to derive project name)"
    )
    trans_parser.add_argument(
        "--lang", "-l",
        help="Target languages, comma-separated (e.g. es,pt,de,fr,ja). If not set, reads from 'languages' field in keywords JSON."
    )
    trans_parser.add_argument(
        "--prompt",
        help="Path to custom translation prompt template (default: built-in)"
    )
    trans_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing translations"
    )
    trans_parser.add_argument(
        "--test", action="store_true",
        help="Test mode: only translate 1 article"
    )

    # ── run (search + collect + generate) ──
    run_parser = subparsers.add_parser(
        "run",
        help="Search, collect, generate (and translate if languages set) in one step"
    )
    run_parser.add_argument(
        "--keywords", "-k", required=True,
        help="Path to keywords JSON file"
    )
    run_parser.add_argument(
        "--output-dir", required=True,
        help="Absolute output root for this run"
    )
    run_parser.add_argument(
        "--prompt",
        help="Path to custom prompt template for generation"
    )
    run_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing articles"
    )
    run_parser.add_argument(
        "--batch-keywords",
        help="Absolute keyword subset file; process one bounded batch under the full --keywords scope"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Derive project name
    if args.command in ("search", "generate", "run"):
        args.project = _derive_project(args.keywords)
        print(f"📁 Project: {args.project}\n")
    elif args.command == "collect":
        if args.project:
            pass
        elif args.keywords:
            args.project = _derive_project(args.keywords)
        else:
            print("❌ Need --project or --keywords to identify project")
            sys.exit(1)
    elif args.command == "translate":
        args.project = _derive_project(args.keywords)
        print(f"📁 Project: {args.project}\n")

    if args.command == "search":
        asyncio.run(_run_search(args))
    elif args.command == "collect":
        asyncio.run(_run_collect(args))
    elif args.command == "generate":
        asyncio.run(_run_generate(args))
    elif args.command == "translate":
        asyncio.run(_run_translate(args))
    elif args.command == "run":
        asyncio.run(_run_all(args))


async def _run_search(args):
    from .search import run_search
    await run_search(args.project, args.keywords)


async def _run_collect(args):
    from .collect import run_collect
    await run_collect(args.project)


async def _run_generate(args):
    from .generate import run_generate
    await run_generate(
        args.project,
        args.keywords,
        prompt_path=args.prompt,
        overwrite=args.overwrite,
        test=args.test,
    )


async def _run_translate(args):
    from .translate import run_translate
    from .core.utils import load_languages_from_json

    # --lang not provided → read from keywords JSON
    if not args.lang:
        langs = load_languages_from_json(args.keywords)
        if not langs:
            print("❌ No --lang specified and no 'languages' field in keywords JSON")
            sys.exit(1)
        args.lang = ",".join(langs)
        print(f"🌐 Languages from JSON: {args.lang}\n")

    await run_translate(
        args.project,
        args.lang,
        prompt_path=args.prompt,
        overwrite=args.overwrite,
        test=args.test,
    )


async def _run_all(args):
    from .search import run_search
    from .collect import run_collect
    from .generate import run_generate
    from .translate import run_translate
    from .core.utils import load_languages_from_json
    from .core.config import Config
    from .contract import (
        ContractError,
        load_keyword_spec,
        language_entries,
        utc_now,
        validate_collected_material,
        validate_english_articles,
        validate_output_dir,
        validate_search_results,
        validate_run_status,
        validate_translated_articles,
        write_run_status,
        write_traceability_manifest,
    )

    try:
        output_root = validate_output_dir(args.output_dir)
    except ContractError as exc:
        print(f"❌ {exc.stage}: {exc.message}", file=sys.stderr)
        sys.exit(exc.exit_code)

    os.environ["OUTPUT_DIR"] = str(output_root)
    Config.init(args.project)
    started_at = utc_now()
    spec = None

    if getattr(args, "batch_keywords", None):
        try:
            spec = load_keyword_spec(args.keywords, Config.DATA_DIR, Config.OUT_DIR)
            await _run_batch(args, spec, output_root)
        except (ContractError, ValueError, OSError) as exc:
            code = exc.exit_code if isinstance(exc, ContractError) else 2
            print(f"❌ Bounded batch stopped ({type(exc).__name__}); original evidence retained", file=sys.stderr)
            sys.exit(code)
        return

    try:
        spec = load_keyword_spec(args.keywords, Config.DATA_DIR, Config.OUT_DIR)
        write_run_status(
            Config.OUT_DIR,
            status="running",
            spec=spec,
            output_root=str(output_root),
            started_at=started_at,
            stage_counts={"expected_keywords": len(spec.entries)},
        )

        try:
            await run_search(args.project, args.keywords)
            validate_search_results(spec, Config.OUT_DIR)
            await run_collect(args.project)
            validate_collected_material(spec)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("search_collect", 3, str(exc)) from exc

        try:
            await run_generate(
                args.project,
                args.keywords,
                prompt_path=args.prompt,
                overwrite=args.overwrite,
                entries=spec.entries,
            )
            validate_english_articles(spec)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("english", 4, str(exc)) from exc

        # If languages are specified in JSON, auto-translate
        langs = load_languages_from_json(args.keywords)
        langs = [lang for lang in langs if lang.lower() != "en"]
        if langs:
            lang_str = ",".join(langs)
            print(f"\n{'='*70}")
            print(f"  Step 4: Translate [{args.project}] → {lang_str}")
            print(f"{'='*70}")
            try:
                await run_translate(
                    args.project,
                    lang_str,
                    prompt_path=None,
                    overwrite=args.overwrite,
                    translation_entries={language: language_entries(spec, language) for language in spec.languages},
                )
                validate_translated_articles(spec)
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError("translation", 5, str(exc)) from exc

        try:
            write_traceability_manifest(spec, Config.OUT_DIR)
            status_path = write_run_status(
                Config.OUT_DIR,
                status="completed",
                spec=spec,
                output_root=str(output_root),
                started_at=started_at,
                stage_counts={
                    "expected_keywords": len(spec.entries),
                    "english_articles": len(spec.entries),
                    "translation_languages": len(spec.languages),
                },
            )
            validate_run_status(status_path, expect_success=True)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("manifest", 6, str(exc)) from exc

    except ContractError as exc:
        if Config.OUT_DIR:
            write_run_status(
                Config.OUT_DIR,
                status="failed",
                spec=spec,
                output_root=str(output_root),
                started_at=started_at,
                failure={
                    "stage": exc.stage,
                    "exit_code": exc.exit_code,
                    "message": exc.message,
                },
            )
        print(f"❌ {exc.stage}: {exc.message}", file=sys.stderr)
        sys.exit(exc.exit_code)


async def _run_batch(args, spec, output_root):
    """One batch in the existing native run; partial never means full completion."""
    from . import contract as c
    from .core.config import Config
    from .search import run_search
    from .collect import run_collect
    from .generate import run_generate
    from .translate import run_translate

    if args.overwrite:
        raise c.ContractError("input", 2, "bounded batches do not overwrite accepted articles")
    selected = c.batch_spec(spec, args.batch_keywords)
    scope = c.batch_scope(spec, selected, output_root, args.prompt)
    lock_path = c.checked_file(output_root / ".seoscout-run.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs = c.verify_batch_history(spec, scope)
        # No API/cache writes for an already accepted batch, even if the whole
        # run is still partial or its HTTP allowance is now exhausted.
        try:
            c.validate_english_articles(selected, spec)
            c.validate_translated_articles(selected, spec)
        except c.ContractError:
            pass
        else:
            if refs:
                print("✅ Batch already accepted; original native run status unchanged")
                return

        budget_path = c.checked_file(Path(Config.LOG_DIR) / "debug/attempts/http-budget.json")
        reserved = 0
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
            if (budget.get("limit") != scope["llm_max_requests"] or budget.get("model") != Config.LLM_MODEL
                    or budget.get("api_url") != Config.LLM_API_BASE_URL.rstrip('/') + '/chat/completions'):
                raise c.ContractError("input", 2, "shared request budget identity mismatch")
            reserved = budget["reserved_requests"]
            for directory in budget.get("attempt_directories", []):
                path = c.checked_file(Path(directory) / "state.json")
                if json.loads(path.read_text()).get("status") in {"pending", "uncertain", "stopped"}:
                    raise c.ContractError("input", 2, "unresolved HTTP attempt; reconcile before batch")
        needed = sum(not entry.english_path.exists() for entry in selected.entries)
        needed += sum(not c._translation_path(entry, language).exists()
                      for language in spec.languages for entry in c.language_entries(selected, language))
        if type(reserved) is not int or reserved < 0 or reserved + needed > scope["llm_max_requests"]:
            raise c.ContractError("input", 2, "remaining HTTP budget cannot cover even the batch's missing pages")

        status_path = Path(Config.OUT_DIR) / "run_status.json"
        if status_path.exists():
            previous = status_path.read_bytes()
            history = c.checked_file(Path(Config.OUT_DIR) / "run-status-history")
            history.mkdir(mode=0o700, exist_ok=True)
            snapshot = history / (hashlib.sha256(previous).hexdigest() + ".json")
            if snapshot.exists():
                if snapshot.read_bytes() != previous:
                    raise c.ContractError("manifest", 6, "status history conflict")
            else:
                descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(previous); stream.flush(); os.fsync(stream.fileno())
        started = c.utc_now()
        counts = {"batch_manifests": refs, "batch_keywords": [entry.keyword for entry in selected.entries]}
        c.write_run_status(Config.OUT_DIR, status="running", spec=spec, output_root=str(output_root),
                           started_at=started, stage_counts=counts, execution_scope=scope)
        previous_scope = os.environ.get("SEOSCOUT_SCOPE_SHA256")
        os.environ["SEOSCOUT_SCOPE_SHA256"] = hashlib.sha256(json.dumps(
            scope, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        def frozen():
            if c.batch_scope(spec, selected, output_root, args.prompt) != scope:
                raise c.ContractError("input", 2, "batch inputs or implementation changed during execution")
        try:
            frozen()
            await run_search(args.project, args.keywords, entries=selected.entries)
            frozen()
            c.validate_search_results(selected, Config.OUT_DIR)
            await run_collect(args.project, entries=selected.entries)
            frozen()
            c.validate_collected_material(selected)
            await run_generate(args.project, args.keywords, prompt_path=args.prompt, entries=selected.entries)
            frozen()
            c.validate_english_articles(selected, spec)
            if spec.languages:
                await run_translate(args.project, ",".join(spec.languages),
                                    translation_entries=selected.translation_entries)
                frozen()
                c.validate_translated_articles(selected, spec)
            c.validate_partial_set(spec)
            manifest = c.build_traceability_manifest(selected, Config.OUT_DIR)
            manifest.update(execution_scope=scope, request_evidence=c.article_request_evidence(selected, scope))
            directory = c.checked_file(Path(Config.OUT_DIR) / "batches")
            directory.mkdir(mode=0o700, exist_ok=True)
            key = hashlib.sha256(json.dumps(counts["batch_keywords"], ensure_ascii=False).encode()).hexdigest()
            path = c.checked_file(directory / (key + ".json"))
            if path.exists():
                raise c.ContractError("manifest", 6, "batch receipt already exists; reconcile instead of overwriting")
            c._atomic_json(path, manifest, exclusive=True)
            refs = [*refs, {"path": str(path), "sha256": c.sha256_file(path)}]
            counts["batch_manifests"] = refs
            complete = all(entry.english_path.is_file() for entry in spec.entries)
            complete = complete and all(c._translation_path(entry, language).is_file()
                                        for language in spec.languages for entry in c.language_entries(spec, language))
            if complete:
                c.validate_english_articles(spec)
                c.validate_translated_articles(spec)
                c.validate_search_results(spec, Config.OUT_DIR)
                c.validate_collected_material(spec)
                final = c.build_traceability_manifest(spec, Config.OUT_DIR)
                final.update(execution_scope=scope, request_evidence=c.article_request_evidence(spec, scope))
                final_path = Path(Config.OUT_DIR) / "traceability_manifest.json"
                if final_path.exists():
                    raise c.ContractError("manifest", 6, "unexpected final manifest; preserve and reconcile")
                c._atomic_json(final_path, final, exclusive=True)
                c.validate_traceability_manifest(final_path)
                counts["final_manifest"] = {"path": str(final_path), "sha256": c.sha256_file(final_path)}
            counts.update(english_articles=sum(entry.english_path.is_file() for entry in spec.entries),
                          translation_articles={language: sum(c._translation_path(entry, language).is_file()
                                                               for entry in c.language_entries(spec, language))
                                                for language in spec.languages})
            c.write_run_status(Config.OUT_DIR, status="completed" if complete else "partial", spec=spec,
                               output_root=str(output_root), started_at=started,
                               stage_counts=counts, execution_scope=scope)
            if complete:
                c.validate_run_status(status_path, expect_success=True)
            print("✅ Full approved collection completed" if complete else "✅ Batch accepted; full run remains partial")
        except Exception as exc:
            code = exc.exit_code if isinstance(exc, c.ContractError) else 6
            c.write_run_status(Config.OUT_DIR, status="failed", spec=spec, output_root=str(output_root),
                               started_at=started, stage_counts=counts, execution_scope=scope,
                               failure={"stage": getattr(exc, "stage", "batch"), "exit_code": code,
                                        "message": "batch stopped; reconcile retained original evidence"})
            raise c.ContractError("batch", code, "batch stopped; original evidence retained") from exc
        finally:
            if previous_scope is None:
                os.environ.pop("SEOSCOUT_SCOPE_SHA256", None)
            else:
                os.environ["SEOSCOUT_SCOPE_SHA256"] = previous_scope


if __name__ == "__main__":
    main()
