"""
LLM API Client — async batch generation with retry logic.

Shared by generate and translate stages. All config from Config class (.env).
"""

import asyncio
import aiohttp
import json
import os
import fcntl
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config


class LLMClient:
    """Async LLM API client with retry, stats, and batch support."""

    def __init__(self):
        self.api_key = Config.LLM_API_KEY
        self.base_url = Config.LLM_API_BASE_URL.rstrip('/')
        self.api_url = f"{self.base_url}/chat/completions"
        self.model = Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS
        self.timeout_seconds = Config.LLM_TIMEOUT
        self.retry_attempts = Config.LLM_RETRY_ATTEMPTS
        self.retry_delay = Config.LLM_RETRY_DELAY

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.stats = {
            'total_requests': 0,
            'http_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'start_time': None,
            'end_time': None,
        }

    # ── single request ──────────────────────────────────────────

    async def generate_single(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        meta: Dict = None,
    ) -> Optional[str]:
        """Send a single prompt, return content string or None."""
        self.stats['total_requests'] += 1
        label = (meta or {}).get('keyword', 'unknown')

        for attempt in range(self.retry_attempts):
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": "You are a professional SEO content writer."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            try:
                attempt_dir = self._reserve_request(payload, meta or {}, attempt + 1)
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"  ❌ Request preflight stopped for {label}: {type(exc).__name__}")
                self.stats['failed_requests'] += 1
                return None
            try:
                self.stats['http_requests'] += 1
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as resp:
                    raw = await resp.read()
                    if self.api_key and self.api_key.encode() in raw:
                        self._record_attempt(attempt_dir, "sensitive_response_omitted", response_sha256=hashlib.sha256(raw).hexdigest())
                        self.stats['failed_requests'] += 1
                        return None
                    self._write_private(attempt_dir / "response.raw", raw)
                    self._record_attempt(attempt_dir, "response_received", http_status=resp.status, response_sha256=hashlib.sha256(raw).hexdigest())
                    if resp.status == 200:
                        data = json.loads(raw)
                        self._save_debug(meta or {}, data)
                        content = data['choices'][0]['message']['content']

                        if 'usage' in data:
                            self.stats['total_tokens'] += data['usage'].get('total_tokens', 0)

                        if not content or not content.strip():
                            print(f"  ⚠️  Empty response for {label} (attempt {attempt+1}/{self.retry_attempts})")
                            if attempt < self.retry_attempts - 1:
                                await asyncio.sleep(self.retry_delay * (attempt + 1))
                                continue
                            self.stats['failed_requests'] += 1
                            return None

                        self.stats['successful_requests'] += 1
                        return content

                    elif resp.status == 429:
                        wait = self.retry_delay * (attempt + 1) * 2
                        print(f"  ⚠️  Rate limited for {label}, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    else:
                        print(f"  ❌ API {resp.status} for {label}; original response saved")
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        self.stats['failed_requests'] += 1
                        return None

            except asyncio.TimeoutError:
                self._record_attempt(attempt_dir, "uncertain", error_class="TimeoutError")
                print(f"  ⏱️  Timeout for {label} (attempt {attempt+1}/{self.retry_attempts})")
                self.stats['failed_requests'] += 1
                return None

            except Exception as e:
                self._record_attempt(attempt_dir, "stopped", error_class=type(e).__name__)
                print(f"  ❌ Exception for {label}: {type(e).__name__}; reconcile before retry")
                self.stats['failed_requests'] += 1
                return None

        self.stats['failed_requests'] += 1
        return None

    # ── batch ───────────────────────────────────────────────────

    async def generate_batch(
        self,
        prompts: List[Tuple[str, Dict]],
        batch_size: int = None,
    ) -> List[Tuple[Dict, Optional[str]]]:
        """
        Batch-generate from list of (prompt, meta) tuples.
        Returns list of (meta, content_or_None) in same order.
        """
        if batch_size is None:
            batch_size = Config.GENERATE_BATCH_SIZE

        self.stats['start_time'] = time.time()
        results: List[Tuple[Dict, Optional[str]]] = []

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(prompts) + batch_size - 1) // batch_size

                print(f"\n  📦 Batch {batch_num}/{total_batches} ({len(batch)} items)...")

                tasks = [
                    self.generate_single(session, prompt, meta)
                    for prompt, meta in batch
                ]
                batch_results = await asyncio.gather(*tasks)

                for (prompt, meta), content in zip(batch, batch_results):
                    results.append((meta, content))

                completed = i + len(batch)
                print(f"  ✅ {completed}/{len(prompts)} done")

                if i + batch_size < len(prompts):
                    await asyncio.sleep(1)

        self.stats['end_time'] = time.time()
        return results

    # ── debug ───────────────────────────────────────────────────

    @staticmethod
    def _write_private(path: Path, data: bytes):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _record_attempt(self, directory: Path, status: str, **details):
        state = {"status": status, "updated_at": time.time(), **details}
        tmp = directory / f"state-{uuid.uuid4().hex}.tmp"
        self._write_private(tmp, json.dumps(state, ensure_ascii=False).encode())
        os.replace(tmp, directory / "state.json")

    def _reserve_request(self, payload: Dict, meta: Dict, attempt: int) -> Path:
        """Persist before HTTP; a shared file caps English, repair and translation POSTs."""
        limit = int(os.getenv("SEOSCOUT_LLM_MAX_REQUESTS", "0"))
        if limit < 0 or not 1 <= self.retry_attempts <= 3:
            raise ValueError("invalid LLM request limits")
        data = json.dumps(payload, ensure_ascii=False).encode()
        if self.api_key and self.api_key.encode() in data:
            raise ValueError("credential present in prompt")
        root = Path(Config.LOG_DIR) / "debug" / "attempts"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        directory = root / uuid.uuid4().hex
        directory.mkdir(mode=0o700)
        self._write_private(directory / "request.json", data)
        self._write_private(directory / "context.json", json.dumps({"meta": meta, "attempt": attempt, "request_sha256": hashlib.sha256(data).hexdigest()}, ensure_ascii=False).encode())
        self._record_attempt(directory, "prepared")
        if limit:
            budget_path = root / "http-budget.json"
            fd = os.open(budget_path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+", encoding="utf-8") as budget_file:
                fcntl.flock(budget_file, fcntl.LOCK_EX)
                content = budget_file.read()
                budget = json.loads(content) if content else {"limit": limit, "reserved_requests": 0, "model": self.model, "api_url": self.api_url}
                if budget.get("limit") != limit or budget.get("model") != self.model or budget.get("api_url") != self.api_url:
                    self._record_attempt(directory, "not_submitted", reason="budget_identity_mismatch")
                    raise RuntimeError("budget identity mismatch")
                for previous in budget.get("attempt_directories", []):
                    previous_state = json.loads((Path(previous) / "state.json").read_text())
                    if previous_state.get("status") in {"pending", "uncertain", "stopped"}:
                        self._record_attempt(directory, "not_submitted", reason="previous_request_unresolved")
                        raise RuntimeError("reconcile previous request before submission")
                if budget["reserved_requests"] >= limit:
                    self._record_attempt(directory, "not_submitted", reason="budget_exhausted")
                    raise RuntimeError("LLM HTTP budget exhausted")
                budget["reserved_requests"] += 1
                budget.setdefault("attempt_directories", []).append(str(directory))
                budget_file.seek(0)
                budget_file.truncate()
                budget_file.write(json.dumps(budget))
                budget_file.flush()
                os.fsync(budget_file.fileno())
        self._record_attempt(directory, "pending")
        return directory

    def _save_debug(self, meta: Dict, data: Dict):
        try:
            debug_dir = Path(Config.LOG_DIR) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            keyword = meta.get('keyword', 'unknown')
            lang = meta.get('language', 'en')
            fp = debug_dir / f"{lang}_{keyword.replace(' ', '_')}_response.json"
            fp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception:
            raise

    # ── stats ───────────────────────────────────────────────────

    def print_stats(self):
        s = self.stats
        duration = (s['end_time'] or time.time()) - (s['start_time'] or time.time())
        rate = s['total_requests'] / duration if duration > 0 else 0
        success_rate = (
            s['successful_requests'] / s['total_requests'] * 100
            if s['total_requests'] > 0 else 0
        )
        print(f"\n  📊 LLM API: {s['successful_requests']}/{s['total_requests']} ok "
              f"({success_rate:.0f}%) | {s['http_requests']} HTTP attempts | {s['total_tokens']} tokens | {duration:.1f}s")
