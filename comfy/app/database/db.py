import glob
import hashlib
import logging
import os
import shutil
from importlib import resources

from ...cli_args import args

Session = None

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    # Use importlib to read alembic.ini from the package
    with resources.as_file(resources.files("comfy") / "alembic.ini") as config_path:
        config = Config(str(config_path))

    # Use module path format for script_location (works with importlib)
    config.set_main_option("script_location", "comfy:alembic_db")
    config.set_main_option("sqlalchemy.url", args.database_url)

    return config


def get_db_path():
    url = args.database_url
    if url.startswith("sqlite:///"):
        return url.split("///")[1]
    else:
        raise ValueError(f"Unsupported database URL '{url}'.")


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


def init_db():
    db_url = args.database_url
    logger.debug(f"Database URL: {db_url}")

    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)
    chain_hash = _compute_chain_hash(script)
    target_rev = script.get_current_head()

    # Derive chain-specific database path
    original_path = db_url.split("///", 1)[1]
    db_dir = os.path.dirname(original_path)
    base, ext = os.path.splitext(os.path.basename(original_path))
    db_path = os.path.join(db_dir, f"{base}-{chain_hash}{ext}")
    db_url = f"sqlite:///{db_path}"
    config.set_main_option("sqlalchemy.url", db_url)

    # If this chain's db doesn't exist, snapshot from the best compatible one
    db_exists = os.path.exists(db_path)
    if not db_exists:
        source = _find_best_source_db(db_dir, base, ext, script, db_path)
        if source:
            logger.info(f"Snapshotting database from '{source}' to '{db_path}'")
            shutil.copy(source, db_path)
            db_exists = True

    engine = create_engine(db_url)
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

    global Session
    Session = sessionmaker(bind=engine)


def create_session():
    return Session()
