"""Durable file replacement.

Every JSON file the pipeline writes is a resume point — downloaded.json,
seed_papers.json, extracted_citations.json, ingested.json — and so is the
pickled BM25 index. A plain ``open(path, "w")`` truncates the old contents
before writing the first byte of the new ones, so a crash, a full disk or a
Ctrl-C partway through leaves a half-written file where a complete one used to
be. Every reader of these files treats unparseable content as "start from
scratch", which turns an interrupted write into a re-download of the whole
corpus.

Writing to a temp file in the *same directory* and then ``os.replace``-ing it
into position makes the swap atomic: a reader sees either the old file or the
new one, never a partial one. Same directory matters — ``os.replace`` is only
atomic within a filesystem.

shared.manifest had this pattern first; this is it extracted so every writer
can share one implementation.
"""

import contextlib
import os
import tempfile


@contextlib.contextmanager
def atomic_write(path: str, binary: bool = False, encoding: str = "utf-8"):
    """Yield a handle whose contents replace *path* only on a clean exit.

    On any exception the temp file is removed and *path* keeps whatever it
    held before, so callers get all-or-nothing semantics without having to
    write their own recovery.

        with atomic_write(MANIFEST_PATH) as fh:
            json.dump(data, fh, indent=2)
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        opener = (
            os.fdopen(fd, "wb")
            if binary
            else os.fdopen(fd, "w", encoding=encoding)
        )
        with opener as handle:
            yield handle
            # Get the bytes onto the platter before the rename, so a power loss
            # cannot leave the new name pointing at an empty file.
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data, **dump_kwargs) -> None:
    """json.dump *data* to *path* atomically. Defaults match the manifests."""
    import json

    dump_kwargs.setdefault("indent", 2)
    with atomic_write(path) as handle:
        json.dump(data, handle, **dump_kwargs)
