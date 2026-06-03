from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from system.core.embeddings import encode_text
from memory.vector_index import VectorIndex


class RagStore:
    def __init__(
        self,
        index_path: str = "artifacts/memory/rag_index.json",
        max_items: int = 8000,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 80,
        max_files: int = 2000,
    ) -> None:
        self.index_path = str(index_path)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.min_chunk_chars = int(min_chunk_chars)
        self.max_files = int(max_files)
        self.index = VectorIndex(dim=384, max_items=max_items)
        self._load()

    def _load(self) -> None:
        try:
            self.index.load(self.index_path)
        except Exception:
            pass

    def save(self) -> None:
        self.index.save(self.index_path)

    def _iter_files(self, path: Path) -> Iterable[Path]:
        if path.is_file():
            yield path
            return
        if not path.exists():
            return
        count = 0
        for p in path.rglob("*"):
            if p.is_file():
                yield p
                count += 1
                if count >= self.max_files:
                    return

    def _allowed_file(self, path: Path) -> bool:
        ext_env = os.environ.get("RAG_EXTS", "").strip()
        if not ext_env:
            return True
        exts = {e.strip().lower() for e in ext_env.split(",") if e.strip()}
        return path.suffix.lower() in exts

    def _read_text(self, path: Path) -> str:
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return path.read_text(encoding=enc)
            except Exception:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = text.replace("\r", "\n")
        cleaned = "\n".join([line.strip() for line in cleaned.splitlines() if line.strip()])
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]
        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(cleaned), step):
            chunk = cleaned[i : i + self.chunk_size]
            if len(chunk) >= self.min_chunk_chars:
                chunks.append(chunk)
        return chunks

    def ingest_paths(self, paths: List[str]) -> int:
        added = 0
        for raw in paths:
            if not raw:
                continue
            p = Path(raw).expanduser()
            for f in self._iter_files(p):
                if not self._allowed_file(f):
                    continue
                try:
                    text = self._read_text(f)
                except Exception:
                    continue
                chunks = self._chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    vec = encode_text(chunk)
                    meta = {"path": str(f), "chunk": idx, "text": chunk}
                    self.index.add(vec, metadata=meta, id=f"{f}:{idx}")
                    added += 1
        return added

    def search(self, query: str, k: int = 4, min_sim: float = 0.2) -> List[Dict[str, Any]]:
        if not query:
            return []
        vec = encode_text(query)
        hits = self.index.search(vec, k=k, threshold=min_sim)
        out: List[Dict[str, Any]] = []
        for _id, score, meta in hits:
            if not isinstance(meta, dict):
                continue
            item = dict(meta)
            item["score"] = float(score)
            out.append(item)
        return out

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200) -> Dict[str, Any] | None:
        hits = self.search(query, k=k, min_sim=min_sim)
        if not hits:
            return None
        snippets: List[Dict[str, Any]] = []
        total = 0
        for h in hits:
            text = str(h.get("text", ""))
            if not text:
                continue
            if total + len(text) > max_chars:
                text = text[: max(0, max_chars - total)]
            if not text:
                break
            snippets.append({
                "text": text,
                "path": h.get("path", ""),
                "score": h.get("score", 0.0),
            })
            total += len(text)
            if total >= max_chars:
                break
        if not snippets:
            return None
        return {"snippets": snippets}

    def export_debug(self, path: str = "artifacts/audit/rag_debug.json") -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                "size": self.index.size(),
                "index_path": self.index_path,
            }
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
