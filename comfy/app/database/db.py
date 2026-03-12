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
# todo: this need merging
import comfy.app.assets.database.models  # noqa: F401 — register models with Base.metadata

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
    db_url = current_execution_context().configuration.database_url
    # Use importlib to read alembic.ini from the package
    with resources.as_file(resources.files("comfy") / "alembic.ini") as config_path:
        config = Config(str(config_path))

    # Use module path format for script_location (works with importlib)
    config.set_main_option("script_location", "comfy:alembic_db")
    config.set_main_option("sqlalchemy.url", db_url)

    return config


def get_db_path():
    url = current_execution_context().configuration.database_url
    if url.startswith("sqlite:///"):
        return url.split("///")[1]
    else:
        raise ValueError(f"Unsupported database URL '{url}'.")


_db_lock = None

def _acquire_file_lock(db_path):
    """Acquire an OS-level file lock to prevent multi-process access.

    Uses filelock for cross-platform support (macOS, Linux, Windows).
    The OS automatically releases the lock when the process exits, even on crashes.
    """
    global _db_lock
    lock_path = db_path + ".lock"
    _db_lock = FileLock(lock_path)
    try:
        _db_lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(
            f"Could not acquire lock on database '{db_path}'. "
            "Another ComfyUI process may already be using it. "
            "Use --database-url to specify a separate database file."
        )


def _is_memory_db(db_url):
    """Check if the database URL refers to an in-memory SQLite database."""
    return db_url in ("sqlite:///:memory:", "sqlite://")


def init_db():
    db_url = current_execution_context().configuration.database_url
    logger.debug(f"Database URL: {db_url}")

    if _is_memory_db(db_url):
        _init_memory_db(db_url)
    else:
        _init_file_db(db_url)


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


def _init_file_db(db_url):
    """Initialize a file-backed SQLite database using Alembic migrations."""
    db_path = get_db_path()
    db_exists = os.path.exists(db_path)

    config = get_alembic_config()

    # Check if we need to upgrade
    engine = create_engine(db_url)

    # Enable foreign key enforcement for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    conn = engine.connect()

    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()

    script = ScriptDirectory.from_config(config)
    target_rev = script.get_current_head()

    if target_rev is None:
        logger.debug("No target revision found.")
    elif current_rev != target_rev:
        # Backup the database pre upgrade
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
                # Restore the database from backup if upgrade fails
                shutil.copy(backup_path, db_path)
                os.remove(backup_path)
            logger.exception("Error upgrading database: ")
            raise e

    # Acquire an OS-level file lock after migrations are complete.
    # Alembic uses its own connection, so we must wait until it's done
    # before locking — otherwise our own lock blocks the migration.
    conn.close()
    _acquire_file_lock(db_path)

    global Session
    Session = sessionmaker(bind=engine)


def create_session():
    return Session()
