import glob
import hashlib
import logging
import os
import shutil
from importlib import resources
from filelock import FileLock, Timeout

from ...execution_context import current_execution_context

Session = None

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base
from ..assets.database import models as _asset_models  # noqa: F401

_DB_AVAILABLE = True

logger = logging.getLogger(__name__)

def dependencies_available():
    """
    Temporary function to check if the dependencies are available
    """
    return _DB_AVAILABLE


def can_create_session():
    """
    Temporary function to check if the database is available to create a session
    During initial release there may be environmental issues (or missing dependencies) that prevent the database from being created
    """
    return dependencies_available() and Session is not None


def get_alembic_config():
    db_url = get_database_url()
    # Use importlib to read alembic.ini from the package
    with resources.as_file(resources.files("comfy") / "alembic.ini") as config_path:
        config = Config(str(config_path))

    # Use module path format for script_location (works with importlib)
    config.set_main_option("script_location", "comfy:alembic_db")
    config.set_main_option("sqlalchemy.url", db_url)

    return config


def get_database_url() -> str:
    config = current_execution_context().configuration
    if config.database_url is not None:
        return config.database_url

    from ...cmd import folder_paths

    db_path = os.path.join(folder_paths.get_user_directory(), "comfyui.db")
    return f"sqlite:///{db_path}"


def get_legacy_default_db_path() -> str:
    from ...cli_args import database_default_path

    return database_default_path


def get_db_path():
    url = get_database_url()
    if url.startswith("sqlite:///"):
        return url.split("///", 1)[1]
    else:
        raise ValueError(f"Unsupported database URL '{url}'.")


def copy_legacy_default_db(db_path: str) -> None:
    config = current_execution_context().configuration
    if config.database_url is not None:
        return

    legacy_db_path = get_legacy_default_db_path()
    if os.path.abspath(legacy_db_path) == os.path.abspath(db_path):
        return
    if os.path.exists(db_path) or not os.path.exists(legacy_db_path):
        return

    backup_path = legacy_db_path + ".bak"
    if os.path.exists(backup_path):
        return

    os.replace(legacy_db_path, backup_path)
    shutil.copy(backup_path, db_path)
    logger.info(
        "Renamed legacy database '%s' to '%s' and copied it to '%s'",
        legacy_db_path,
        backup_path,
        db_path,
    )


def prepare_file_db_path(db_path: str) -> None:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    copy_legacy_default_db(db_path)


_db_lock = None

def _acquire_file_lock(db_path) -> bool:
    """Try to acquire an OS-level file lock. Returns True on success, False on contention."""
    global _db_lock
    lock_path = db_path + ".lock"
    _db_lock = FileLock(lock_path)
    try:
        _db_lock.acquire(timeout=0)
        return True
    except Timeout:
        return False


def _is_memory_db(db_url):
    """Check if the database URL refers to an in-memory SQLite database."""
    return db_url in ("sqlite:///:memory:", "sqlite://")


def init_db():
    config = current_execution_context().configuration
    db_url = get_database_url()
    explicit = getattr(config, "_database_url_explicit", False)
    logger.debug(f"Database URL: {db_url} (explicit={explicit})")

    if _is_memory_db(db_url):
        _init_memory_db(db_url)
    else:
        _init_file_db(db_url, use_chain_hash=not explicit)


def _init_memory_db(db_url):
    """Initialize an in-memory SQLite database using metadata.create_all.

    Alembic migrations don't work with in-memory SQLite because each
    connection gets its own separate database — tables created by Alembic's
    internal connection are lost immediately.
    """
    engine = create_engine(
        db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    global Session
    Session = sessionmaker(bind=engine)


def _compute_chain_hash(script: ScriptDirectory) -> str:
    """Compute a short hash of the ordered migration revision chain.

    Each revision depends on its parent, so the hash captures the full
    chain merkle-style: if any revision is added, removed, or reordered,
    the hash changes, producing a distinct database file.
    """
    revisions = list(script.walk_revisions())  # head-to-base order
    rev_ids = [r.revision for r in reversed(revisions)]  # base-to-head
    chain_str = ":".join(rev_ids)
    return hashlib.sha256(chain_str.encode()).hexdigest()[:12]


def _find_best_source_db(db_dir, db_basename, db_ext, script, exclude_path):
    """Find the most recent compatible database file to snapshot from.

    Searches for existing database files whose alembic revision is a
    known ancestor in our migration chain, and returns the one that is
    furthest along.
    """
    revisions = list(script.walk_revisions())
    rev_order = {r.revision: i for i, r in enumerate(reversed(revisions))}

    candidates = glob.glob(os.path.join(db_dir, f"{db_basename}{db_ext}"))
    candidates += glob.glob(os.path.join(db_dir, f"{db_basename}-*{db_ext}"))

    best_path = None
    best_position = -1

    for db_file in candidates:
        if db_file == exclude_path or not os.path.isfile(db_file):
            continue
        try:
            eng = create_engine(f"sqlite:///{db_file}")
            with eng.connect() as conn:
                ctx = MigrationContext.configure(conn)
                rev = ctx.get_current_revision()
            eng.dispose()
        except Exception:
            continue

        if rev is not None and rev in rev_order and rev_order[rev] > best_position:
            best_position = rev_order[rev]
            best_path = db_file

    return best_path


def _init_file_db(db_url, use_chain_hash: bool = True):
    """Initialize a file-backed SQLite database using Alembic migrations.

    When use_chain_hash is True (default, for auto-generated URLs), the
    database filename includes a hash of the migration revision chain so
    each distinct set of migrations gets its own file.

    When use_chain_hash is False (user-provided --database-url), the URL
    is used exactly as given.

    If the file lock cannot be acquired, falls back to an in-memory database
    so multiple instances can coexist without crashing.
    """
    original_path = db_url.split("///", 1)[1]
    prepare_file_db_path(original_path)

    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)
    target_rev = script.get_current_head()

    if use_chain_hash:
        chain_hash = _compute_chain_hash(script)
        db_dir = os.path.dirname(original_path)
        base, ext = os.path.splitext(os.path.basename(original_path))
        db_path = os.path.join(db_dir, f"{base}-{chain_hash}{ext}")
        db_url = f"sqlite:///{db_path}"

        db_exists = os.path.exists(db_path)
        if not db_exists:
            source = _find_best_source_db(db_dir, base, ext, script, db_path)
            if source:
                logger.info(f"Snapshotting database from '{source}' to '{db_path}'")
                shutil.copy(source, db_path)
                db_exists = True
    else:
        db_path = original_path
        db_exists = os.path.exists(db_path)

    config.set_main_option("sqlalchemy.url", db_url)

    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    conn = engine.connect()
    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()

    if target_rev is None:
        logger.debug("No target revision found.")
    elif current_rev != target_rev:
        backup_path = db_path + ".bkp"
        if db_exists:
            shutil.copy(db_path, backup_path)
        else:
            backup_path = None

        try:
            command.upgrade(config, target_rev)
            logger.info(f"Database upgraded from {current_rev} to {target_rev}")
        except Exception as e:
            if backup_path:
                shutil.copy(backup_path, db_path)
                os.remove(backup_path)
            logger.exception("Error upgrading database: ")
            raise e

    conn.close()

    if not _acquire_file_lock(db_path):
        engine.dispose()
        logger.warning(
            f"Database '{db_path}' is locked by another process. "
            "Falling back to in-memory database for this instance."
        )
        _init_memory_db("sqlite:///:memory:")
        return

    global Session
    Session = sessionmaker(bind=engine)


def create_session():
    return Session()
