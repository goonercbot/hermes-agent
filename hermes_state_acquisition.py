"""Bound SQLite acquisition independently of a caller's statement busy timeout."""

import logging

logger = logging.getLogger(__name__)


def begin_immediate(conn):
    """Python's retry deadline owns waiting; restore before running user statements.

    A failed restoration after acquisition must roll back before a callback can
    run. Never replace the original acquisition error if restoration also fails.
    """
    previous = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    conn.execute("PRAGMA busy_timeout=0")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except BaseException:
        try:
            conn.execute(f"PRAGMA busy_timeout={previous}")
        except BaseException:
            logger.debug("Could not restore busy timeout after failed BEGIN", exc_info=True)
        raise
    try:
        conn.execute(f"PRAGMA busy_timeout={previous}")
    except BaseException:
        try:
            conn.rollback()
        except BaseException:
            logger.debug("Could not roll back after timeout restoration failure", exc_info=True)
        raise
