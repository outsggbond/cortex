from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple, List


class FileManager:
    def __init__(self, root: str, backup_dir: str = ".rollback"):
        self.root = Path(root).resolve()
        self.backup_dir = (self.root / backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path outside project root")
        return p

    def read_text(self, path: str) -> str:
        p = self._resolve(path)
        return p.read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # backup
        if p.exists():
            rel = p.relative_to(self.root).as_posix()
            safe = rel.replace("/", "__")
            backup = self.backup_dir / f"{safe}.bak"
            shutil.copy2(p, backup)
        p.write_text(content, encoding="utf-8")
        return str(p)

    def append_text(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return str(p)

    def touch(self, path: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("", encoding="utf-8")
        return str(p)

    def list_dir(self, path: str = ".") -> List[str]:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"dir not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"not a dir: {p}")
        return sorted([c.name for c in p.iterdir()])

    def exists(self, path: str) -> bool:
        p = self._resolve(path)
        return p.exists()

    def mkdir(self, path: str) -> str:
        p = self._resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def copy(self, src: str, dst: str) -> Tuple[str, str]:
        s = self._resolve(src)
        d = self._resolve(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        return str(s), str(d)

    def move(self, src: str, dst: str) -> Tuple[str, str]:
        s = self._resolve(src)
        d = self._resolve(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return str(s), str(d)

    def delete(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"path not found: {p}")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return str(p)

    def rollback(self, path: str) -> bool:
        p = self._resolve(path)
        rel = p.relative_to(self.root).as_posix()
        safe = rel.replace("/", "__")
        backup = self.backup_dir / f"{safe}.bak"
        if backup.exists():
            shutil.copy2(backup, p)
            return True
        return False
