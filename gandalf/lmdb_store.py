"""LMDB-backed edge property storage for publications, sources, and attributes.

These are the "cold path" properties — only accessed during response enrichment
for the small set of result edges. Qualifiers and sources (hot path, needed during
traversal filtering) are kept in-memory via the dedup store in graph.py.

LMDB is a memory-mapped B-tree. Multiple worker processes share the same physical
memory pages via the OS, identical to how numpy mmap works. No per-process
duplication.
"""

import shutil
import struct
import tempfile
from pathlib import Path

import lmdb
import msgpack


# 50 GB virtual address space (not allocated until used)
_DEFAULT_MAP_SIZE = 50 * 1024 * 1024 * 1024


def _encode_key(edge_idx: int) -> bytes:
    """Encode edge index as 4-byte big-endian for correct LMDB sort order."""
    return struct.pack(">I", edge_idx)


def _decode_key(key: bytes) -> int:
    """Decode 4-byte big-endian key back to edge index."""
    return struct.unpack(">I", key)[0]


class LMDBPropertyStore:
    """Disk-backed edge property storage using LMDB.

    Stores publications, sources, and attributes per edge as msgpack blobs.
    Keys are edge indices (matching CSR array positions) encoded as 4-byte
    big-endian integers for correct sort order.

    Memory-mapped by the OS — multiple worker processes reading the same
    file share physical memory pages automatically.
    """

    def __init__(self, path, readonly=True):
        self._path = Path(path)
        self._env = lmdb.open(
            str(self._path),
            readonly=readonly,
            max_dbs=0,
            map_size=_DEFAULT_MAP_SIZE,
            readahead=False,  # We do point + small range reads
            lock=not readonly,  # No lock file needed for read-only
        )

    def get(self, edge_idx):
        """Get all detail properties for a single edge.

        Returns dict with 'publications', 'sources', 'attributes' keys,
        or empty dict if edge not found.
        """
        key = _encode_key(edge_idx)
        with self._env.begin(buffers=True) as txn:
            val = txn.get(key)
            if val is None:
                return {}
            return msgpack.unpackb(val, raw=False)

    def get_batch(self, edge_indices):
        """Get detail properties for multiple edges.

        Used during response enrichment for the result set (typically
        tens to low hundreds of edges).

        Returns dict mapping edge_idx -> properties dict.
        """
        results = {}
        with self._env.begin(buffers=True) as txn:
            for idx in edge_indices:
                key = _encode_key(idx)
                val = txn.get(key)
                if val is not None:
                    results[idx] = msgpack.unpackb(val, raw=False)
        return results

    def close(self):
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    @staticmethod
    def build(db_path, edge_iterator, num_edges, commit_every=50_000):
        """Build an LMDB store by streaming edge properties.

        Args:
            db_path: Path for the LMDB directory.
            edge_iterator: Yields (edge_idx, props_dict) tuples where
                props_dict has keys 'publications', 'sources', 'attributes'.
                Must yield in edge_idx order (0, 1, 2, ...).
            num_edges: Total number of edges (for progress reporting).
            commit_every: Commit transaction every N edges to limit memory.

        Returns:
            LMDBPropertyStore opened in read-only mode.
        """
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            str(db_path),
            map_size=_DEFAULT_MAP_SIZE,
            readonly=False,
            max_dbs=0,
            readahead=False,
        )

        txn = env.begin(write=True)
        count = 0
        try:
            for edge_idx, props in edge_iterator:
                key = _encode_key(edge_idx)
                val = msgpack.packb(props, use_bin_type=True)
                txn.put(key, val)
                count += 1

                if count % commit_every == 0:
                    txn.commit()
                    if count % 1_000_000 == 0:
                        print(f"    LMDB: wrote {count:,}/{num_edges:,} edges...")
                    txn = env.begin(write=True)

            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            env.close()

        print(f"    LMDB: wrote {count:,} edges to {db_path}")
        return LMDBPropertyStore(db_path, readonly=True)

    @staticmethod
    def build_sorted(db_path, temp_db_path, sort_permutation, num_edges,
                     commit_every=50_000):
        """Rewrite a temp LMDB in CSR-sorted order to produce the final store.

        Reads from temp_db_path using sort_permutation to reorder, writes
        to db_path with sequential keys 0..num_edges-1.

        This is the expensive build-time operation that ensures query-time
        keys match CSR edge indices with zero indirection.

        Args:
            db_path: Path for the final LMDB directory.
            temp_db_path: Path to temporary LMDB (keyed by original line index).
            sort_permutation: numpy array where sort_permutation[csr_pos] = original_line_idx.
            num_edges: Total number of edges.
            commit_every: Commit transaction every N edges.

        Returns:
            LMDBPropertyStore opened in read-only mode.
        """
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        temp_env = lmdb.open(
            str(temp_db_path),
            readonly=True,
            lock=False,
            map_size=_DEFAULT_MAP_SIZE,
            readahead=False,
        )
        final_env = lmdb.open(
            str(db_path),
            map_size=_DEFAULT_MAP_SIZE,
            readonly=False,
            max_dbs=0,
            readahead=False,
        )

        print(f"    LMDB: rewriting {num_edges:,} edges in CSR-sorted order...")

        temp_txn = temp_env.begin(buffers=True)
        final_txn = final_env.begin(write=True)

        try:
            for csr_pos in range(num_edges):
                original_idx = int(sort_permutation[csr_pos])
                temp_key = _encode_key(original_idx)
                val = temp_txn.get(temp_key)

                final_key = _encode_key(csr_pos)
                final_txn.put(final_key, bytes(val))

                if (csr_pos + 1) % commit_every == 0:
                    final_txn.commit()
                    if (csr_pos + 1) % 1_000_000 == 0:
                        print(f"      {csr_pos + 1:,}/{num_edges:,} edges rewritten...")
                    final_txn = final_env.begin(write=True)

            final_txn.commit()
        except BaseException:
            final_txn.abort()
            raise
        finally:
            temp_txn.abort()
            temp_env.close()
            final_env.close()

        print(f"    LMDB: final store written to {db_path}")
        return LMDBPropertyStore(db_path, readonly=True)
