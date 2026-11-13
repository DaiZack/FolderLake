"""Local registry for the FolderLake app (home page).

A small SQLite database — stdlib only — that remembers every catalog
the app has built (so the home page can list history and reopen one in
a click) and the user's settings (LLM provider/model, defaults). Each
catalog record also stores a snapshot of the settings its last scan
ran with, so the history shows *how* a catalog was generated, not just
where it lives.

Location: ``$FOLDERLAKE_HOME/registry.sqlite3``, defaulting to
``~/.folderlake/registry.sqlite3``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

SETTING_KEYS = ("llm_provider", "llm_model", "llm_base_url",
                "llm_api_key", "use_llm")


def registry_home() -> Path:
    home = Path(os.environ.get("FOLDERLAKE_HOME",
                               Path.home() / ".folderlake"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def catalog_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]


class Registry:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (registry_home() / "registry.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path),
                                    check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS catalogs (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_opened_at REAL,
                file_count INTEGER DEFAULT 0,
                catalog_bytes INTEGER DEFAULT 0,
                settings TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._con.commit()

    # -- catalogs ----------------------------------------------------
    def list_catalogs(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT * FROM catalogs ORDER BY updated_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["settings"] = json.loads(d["settings"] or "{}")
            d["exists"] = Path(d["path"]).is_dir()
            out.append(d)
        return out

    def get_catalog(self, catalog_id: str) -> dict | None:
        row = self._con.execute(
            "SELECT * FROM catalogs WHERE id = ?", [catalog_id]).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["settings"] = json.loads(d["settings"] or "{}")
        d["exists"] = Path(d["path"]).is_dir()
        return d

    def upsert_catalog(self, path: Path, file_count: int,
                       catalog_bytes: int, settings: dict) -> dict:
        path = path.resolve()
        cid = catalog_id_for(path)
        now = time.time()
        self._con.execute(
            """
            INSERT INTO catalogs (id, path, name, created_at, updated_at,
                                  file_count, catalog_bytes, settings)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                updated_at=excluded.updated_at,
                file_count=excluded.file_count,
                catalog_bytes=excluded.catalog_bytes,
                settings=excluded.settings
            """,
            [cid, str(path), path.name, now, now, file_count,
             catalog_bytes, json.dumps(settings, ensure_ascii=False)],
        )
        self._con.commit()
        return self.get_catalog(cid)  # type: ignore[return-value]

    def touch_opened(self, catalog_id: str) -> None:
        self._con.execute(
            "UPDATE catalogs SET last_opened_at = ? WHERE id = ?",
            [time.time(), catalog_id])
        self._con.commit()

    def delete_catalog(self, catalog_id: str) -> dict | None:
        rec = self.get_catalog(catalog_id)
        if rec is None:
            return None
        self._con.execute("DELETE FROM catalogs WHERE id = ?",
                          [catalog_id])
        self._con.commit()
        return rec

    # -- settings ------------------------------------------------------
    def get_settings(self) -> dict:
        rows = self._con.execute("SELECT key, value FROM settings").fetchall()
        saved = {r["key"]: r["value"] for r in rows}
        return {
            "llm_provider": saved.get("llm_provider", "ollama"),
            "llm_model": saved.get("llm_model", ""),
            "llm_base_url": saved.get("llm_base_url", ""),
            "llm_api_key_set": bool(saved.get("llm_api_key")),
            "use_llm": saved.get("use_llm", "0") == "1",
        }

    def get_api_key(self) -> str:
        row = self._con.execute(
            "SELECT value FROM settings WHERE key = 'llm_api_key'"
        ).fetchone()
        return row["value"] if row and row["value"] else ""

    def save_settings(self, values: dict) -> dict:
        for key in SETTING_KEYS:
            if key not in values:
                continue
            v = values[key]
            if key == "llm_api_key" and (v is None or v == ""):
                continue  # blank field = keep the saved key
            if key == "use_llm":
                v = "1" if v else "0"
            self._con.execute(
                "INSERT INTO settings (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [key, "" if v is None else str(v)])
        self._con.commit()
        return self.get_settings()
