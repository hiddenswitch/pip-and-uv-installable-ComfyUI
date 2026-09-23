from importlib.resources import files
from unittest.mock import Mock
import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config
from filelock import FileLock, Timeout

from comfy.app.database import db as db_module
from comfy.cmd import main

_PRE_HEAD = "0006_add_loader_path"


def _make_config(db_path: str) -> Config:
    resources = files("comfy")
    cfg = Config(str(resources.joinpath("alembic.ini")))
    cfg.set_main_option("script_location", str(resources.joinpath("alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _current_revision(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
    assert len(rows) == 1
    return rows[0][0]


@pytest.fixture
def stale_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "comfyui.db")
    command.upgrade(_make_config(db_path), _PRE_HEAD)

    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield db_path
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def test_init_file_db_migrates_when_lock_is_free(stale_db):
    db_module._init_file_db(f"sqlite:///{stale_db}", use_chain_hash=False)

    assert _current_revision(stale_db) != _PRE_HEAD
    assert os.path.exists(stale_db + ".bkp")


def test_successful_init_keeps_holding_the_lock(stale_db):
    db_module._init_file_db(f"sqlite:///{stale_db}", use_chain_hash=False)

    contender = FileLock(stale_db + ".lock")
    with pytest.raises(Timeout):
        contender.acquire(timeout=0)


def test_failed_init_releases_the_lock(stale_db, monkeypatch):
    def _explode(*_args):
        raise RuntimeError("alembic config exploded")

    monkeypatch.setattr(db_module, "_migrate_and_bind", _explode)

    with pytest.raises(RuntimeError, match="alembic config exploded"):
        db_module._init_file_db(f"sqlite:///{stale_db}", use_chain_hash=False)

    contender = FileLock(stale_db + ".lock")
    try:
        contender.acquire(timeout=0)
    except Timeout:
        pytest.fail(
            "a failed init stranded the lock; setup_database logs and CONTINUES when assets are "
            "disabled, so this process would block every other instance for its whole lifetime "
            "over a database it never opened"
        )
    contender.release()


def test_held_lock_blocks_before_any_migration_work(stale_db):
    holder = FileLock(stale_db + ".lock")
    holder.acquire(timeout=0)
    try:
        db_module._init_file_db(f"sqlite:///{stale_db}", use_chain_hash=False)

        assert not os.path.exists(stale_db + ".bkp")
        assert _current_revision(stale_db) == _PRE_HEAD
        assert db_module.Session is not None
        assert db_module.Session.kw["bind"].url.database == ":memory:"
    finally:
        holder.release()


def test_setup_database_starts_assets_after_database_init(monkeypatch):
    events = []
    monkeypatch.setattr(db_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(db_module, "init_db", lambda: events.append("database"))
    manager = Mock()
    manager.startup.side_effect = lambda: events.append("assets")
    main.setup_database(manager)
    assert events == ["database", "assets"]


def test_setup_database_does_not_start_assets_after_database_failure(monkeypatch):
    monkeypatch.setattr(db_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(db_module, "init_db", Mock(side_effect=RuntimeError("migration failed")))
    manager = Mock()
    with pytest.raises(RuntimeError, match="migration failed"):
        main.setup_database(manager)
    manager.startup.assert_not_called()
