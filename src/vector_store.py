import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

import ollama
import json

from config import config


META_KEYS = ("source", "source_path", "filename", "chunk_index", "total_chunks")


def _chunk_order(rows) -> int:
    """Sort key: order chunks by their chunk_index for a stable read."""
    meta = rows[2]
    return (meta or {}).get("chunk_index", 0)


class VectorStore:
    """A single database, backed by a ChromaDB *collection*.

    A "database" in this app is a named collection living inside the one
    persistent store at ``config.chroma_path``. Multiple databases therefore
    coexist in the same store, which means switching, renaming, or creating one
    database never touches the data in another.

    Pass a ``collection_name`` explicitly to open a specific database, or omit
    it to open the active database (``config.collection_name``).
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.client = chromadb.PersistentClient(
            path=config.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        name = _sanitize_name(collection_name or config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = name
        self.ollama_client = ollama.Client(host=config.ollama_host)

    # ------------------------------------------------------------------ embed
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.ollama_client.embeddings(
                model=config.embedding_model, prompt=text,
            )
            embeddings.append(response["embedding"])
        return embeddings

    def _get_embedding(self, text: str) -> List[float]:
        response = self.ollama_client.embeddings(
            model=config.embedding_model, prompt=text,
        )
        return response["embedding"]

    # ------------------------------------------------------------- read/write
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        raw = self._get_embeddings_batch([doc["text"] for doc in documents])

        seen = set()
        texts, metadatas, ids, embeddings = [], [], [], []
        for doc, emb in zip(documents, raw):
            id_ = f"{doc['source']}#{doc['chunk_index']}"
            if id_ in seen:
                continue
            seen.add(id_)
            texts.append(doc["text"])
            metadatas.append(
                {k: doc.get(k) for k in META_KEYS if doc.get(k) is not None}
            )
            ids.append(id_)
            embeddings.append(emb)

        if not ids:
            return

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def search(self, query: str, n_results: int = 5,
               where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = results["documents"]
        if not docs or not docs[0]:
            return []

        return [
            {
                "text": docs[0][i],
                "metadata": (results["metadatas"][0][i]
                             if results["metadatas"][0] else None),
                "distance": (results["distances"][0][i]
                             if results["distances"] and results["distances"][0]
                             else None),
            }
            for i in range(len(docs[0]))
        ]

    def get_all_sources(self) -> List[str]:
        results = self.collection.get()
        if not results["metadatas"]:
            return []
        return sorted({m["source"] for m in results["metadatas"] if m})

    def get_source_counts(self) -> Dict[str, int]:
        results = self.collection.get()
        counts: Dict[str, int] = {}
        for m in results["metadatas"] or []:
            if m:
                counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def get_source(self, source_key: str,
                   collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Fetch every chunk for one source in one database."""
        col = self._col(collection_name or self.collection_name)
        results = col.get(where={"source": source_key})
        rows = list(zip(
            results.get("ids") or [],
            results.get("documents") or [],
            results.get("metadatas") or [],
        ))
        rows.sort(key=_chunk_order)

        chunks, ids = [], []
        for id_, doc, meta in rows:
            m = meta or {}
            chunks.append({
                "id": id_,
                "text": doc or "",
                "source": m.get("source"),
                "source_path": m.get("source_path"),
                "filename": m.get("filename"),
                "chunk_index": m.get("chunk_index"),
                "total_chunks": m.get("total_chunks"),
            })
            if id_:
                ids.append(id_)
        return {
            "found": bool(chunks), "chunks": chunks, "ids": ids,
            "meta": rows[0][2] if rows else None,
        }

    def delete_source(self, source_key: str,
                      collection_name: Optional[str] = None) -> int:
        col = self._col(collection_name or self.collection_name)
        results = col.get(where={"source": source_key})
        if results["ids"]:
            col.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def get_stats(self) -> Dict[str, Any]:
        results = self.collection.get()
        total_chunks = len(results["ids"]) if results["ids"] else 0
        sources = {m["source"] for m in results["metadatas"] or []
                   if m and m.get("source") is not None}
        return {
            "total_chunks": total_chunks,
            "unique_sources": len(sources),
            "sources": sorted(sources),
            "collection": self.collection_name,
        }

    def clear(self) -> None:
        """Clear all documents from *this* database only."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )

    def _col(self, name: str):
        return self.client.get_or_create_collection(
            name=_sanitize_name(name), metadata={"hnsw:space": "cosine"},
        )


def _sanitize_name(name: str) -> str:
    """Make a user-typed database name a valid ChromaDB collection name."""
    out = [c if (c.isalnum() or c in "_-") else "_" for c in str(name).strip()]
    name = "".join(out).strip("_-")
    return name or "db"


def _load_state(state_file: str) -> Dict[str, Any]:
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state_file: str, data: Dict[str, Any]) -> None:
    try:
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


class DatabaseManager:
    """Manage multiple databases (ChromaDB collections) in one persistent store.

    This is the control layer the CLI and TUI use to create, inspect, switch
    between, rename, copy, reassign, and delete databases without ever wiping
    one in order to operate on another.
    """

    def __init__(self, chroma_path: Optional[str] = None,
                 state_file: Optional[str] = None):
        self.chroma_path = chroma_path or config.chroma_path
        self.state_file = state_file or config.state_file
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

    # --------------------------------------------------------- active database
    def get_active(self) -> str:
        active = _load_state(self.state_file).get("active")
        if active and self._exists(_sanitize_name(active)):
            return _sanitize_name(active)
        active = _sanitize_name(config.collection_name)
        self.set_active(active)
        return active

    def ensure(self, name: str) -> str:
        name = _sanitize_name(name)
        self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"},
        )
        return name

    def set_active(self, name: str) -> str:
        name = self.ensure(name)
        state = _load_state(self.state_file)
        state["active"] = name
        # Keep the global config in sync so fresh VectorStore / DocumentChat
        # instances created later in this process use the same database.
        config.collection_name = name
        _save_state(self.state_file, state)
        return name

    # -------------------------------------------------------------- listings
    def _exists(self, name: str) -> bool:
        return any(c.name == name for c in self.client.list_collections())

    def _col(self, name: str):
        return self.client.get_or_create_collection(
            name=_sanitize_name(name), metadata={"hnsw:space": "cosine"},
        )

    def list_databases(self) -> List[Dict[str, Any]]:
        active = self.get_active()
        entries = []
        for col in self.client.list_collections():
            res = col.get()
            metas = res.get("metadatas") or []
            counts: Dict[str, int] = {}
            for m in metas:
                if m:
                    counts[m.get("source")] = counts.get(m.get("source"), 0) + 1
            ids = res.get("ids") or []
            entries.append({
                "name": col.name,
                "chunks": len(ids),
                "sources": sorted(k for k in counts),
                "source_counts": counts,
                "active": col.name == active,
            })
        entries.sort(key=lambda e: e["name"])
        return entries

    # --------------------------------------------------------- create / delete
    def create(self, name: str) -> str:
        name = self.ensure(name)
        return name

    def delete(self, name: str) -> bool:
        name = _sanitize_name(name)
        if not self._exists(name):
            return False
        self.client.delete_collection(name)
        # Fall back to a sane active database after removal if needed.
        if _load_state(self.state_file).get("active") == name:
            state = _load_state(self.state_file)
            fallback = _sanitize_name(config.collection_name) if config.collection_name != name else "documents"
            state["active"] = fallback
            config.collection_name = fallback
            _save_state(self.state_file, state)
        return True

    def rename(self, old: str, new: str) -> str:
        old_n = _sanitize_name(old)
        new_n = _sanitize_name(new)
        if old_n == new_n:
            return new_n
        if not self._exists(old_n):
            raise ValueError(f"No database named {old!r}")
        src = self._col(old_n)
         # get() with the default include returns ids + documents + metadatas
         # but embeddings=None.  Fetch embeddings in a second, id-scoped call so
         # we never lose the documents/metadatas or re-embed.
        res = src.get()
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        if ids:
            emb_res = src.get(ids=ids, include=["embeddings"])
            embs = emb_res.get("embeddings")
            embs = embs if embs is not None else []
            self._col(new_n).add(
                documents=docs,
                metadatas=metas,
                ids=ids,
                embeddings=embs,
              )
        self.client.delete_collection(old_n)
        if _load_state(self.state_file).get("active") == old_n:
            self.set_active(new_n)
        return new_n

    # -------------------------------------------------------------- copy / move
    def copy_source(self, source_key: str, dest: str,
                    src_db: Optional[str] = None,
                    remove_source: Optional[bool] = None) -> Dict[str, Any]:
        """Move or copy one document from one database to another.

        ``dest`` is a destination database name; if it does not yet exist it is
        created, so a document can be added to a brand-new database without
        touching any existing one. The copy carries the stored embeddings, so no
        re-embedding happens.
        """
        dest_n = self.ensure(dest)
        src_db = _sanitize_name(src_db or self.get_active())
        if not self._exists(src_db):
            raise ValueError(f"No source database named {src_db!r}")

        chunks = self.collection_chunks(src_db, source_key)
        if not chunks:
            raise ValueError(f"No document {source_key!r} in database {src_db!r}")

        added = self._copy_chunks_into(src_db, dest_n, chunks)

        removed = 0
         # remove_source is None for the default "move" behaviour,
         # False to keep the source, True to always remove.
        if remove_source is None:
            remove_source = (src_db != dest_n)
        if remove_source:
            removed = self.collection_delete_source(src_db, source_key)

        return {
            "source_key": source_key,
            "from": src_db,
            "to": dest_n,
            "chunks_added": added,
            "removed_from_source": removed,
            "dest_database": dest_n,
         }

    def _copy_chunks_into(self, src_db: str, dest_db: str,
                          chunks: List[Dict[str, Any]]) -> int:
        ids_in = [c["id"] for c in chunks]
        got = self._col(src_db).get(ids=ids_in, include=["embeddings"])
        got_ids = got.get("ids")
        got_embs = got.get("embeddings")
        got_embs = got_embs if got_embs is not None else []
        embed = dict(zip(got_ids or [], got_embs))

        dst = self._col(dest_db)
        existing_res = dst.get(ids=ids_in)
        existing = set(existing_res.get("ids") or [])

        ids, docs, metas, embs = [], [], [], []
        for c in chunks:
            if c["id"] in existing or c["id"] not in embed:
                continue
            ids.append(c["id"])
            docs.append(c["text"])
            embs.append(embed[c["id"]])
            metas.append({k: c.get(k) for k in META_KEYS if c.get(k) is not None})

        if ids:
            dst.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        return len(ids)

    # ------------------------------------------------------ internal helpers
    def collection_chunks(self, collection_name: str,
                          source_key: str) -> List[Dict[str, Any]]:
        res = self._col(collection_name).get(where={"source": source_key})
        rows = list(zip(
            res.get("ids") or [],
            res.get("documents") or [],
            res.get("metadatas") or [],
        ))
        rows.sort(key=_chunk_order)
        out = []
        for id_, doc, meta in rows:
            m = meta or {}
            out.append({
                "id": id_,
                "text": doc or "",
                "source": m.get("source"),
                "source_path": m.get("source_path"),
                "filename": m.get("filename"),
                "chunk_index": m.get("chunk_index"),
                "total_chunks": m.get("total_chunks"),
            })
        return out

    def collection_delete_source(self, collection_name: str,
                                 source_key: str) -> int:
        res = self._col(collection_name).get(where={"source": source_key})
        if res["ids"]:
            self._col(collection_name).delete(ids=res["ids"])
            return len(res["ids"])
        return 0


def main():
    store = VectorStore()
    stats = store.get_stats()
    print(f"Database: {stats['collection']}    "
          f"{stats['total_chunks']} chunks, "
          f"{stats['unique_sources']} documents")
    for s in stats["sources"]:
        print(f"          -> {s}")


if __name__ == '__main__':
    main()
