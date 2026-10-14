"""Episode traces + late immutable reward; no dense pair-state archives."""

import json, sqlite3
import torch
from dataclasses import asdict
from ..contracts import d_path, BuildEpisode, BuildAction, GeneratedProtein, digest
from ..geometry.ca_builder import replay


class EpisodeReplay:
    def __init__(self, path):
        self.db = sqlite3.connect(d_path(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL,reward TEXT)")

    def store(self, episode, generated_protein=None):
        record = {
            "N": episode.N,
            "behavior": episode.behavior,
            "actions": [
                {"residue": a.residue, "geometry": a.geometry.detach().cpu().tolist(), "mask": a.mask}
                for a in episode.actions
            ],
        }
        if generated_protein is not None:
            reconstructed = GeneratedProtein(replay(episode.N, episode.actions))
            if generated_protein.sequence_sha256 != reconstructed.sequence_sha256:
                raise ValueError("stored proposal sequence differs from replay")
            # Exact proposal bytes remain authoritative; roundtrip tolerance only checks trace correspondence.
            torch.testing.assert_close(
                generated_protein.state.ca.detach().cpu().double(),
                reconstructed.state.ca,
                rtol=1e-8 if generated_protein.state.ca.dtype == torch.float64 else 1e-4,
                atol=1e-8 if generated_protein.state.ca.dtype == torch.float64 else 1e-4,
            )
            record.update(
                proposal_ca=generated_protein.state.ca.detach().cpu().tolist(),
                proposal_sha256=generated_protein.geometry_sha256,
                geometry_version=generated_protein.state.geometry_version,
            )
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        old = self.db.execute("SELECT payload FROM episodes WHERE id=?", (episode.episode_id,)).fetchone()
        if old and old[0] != payload:
            raise ValueError("immutable episode mismatch")
        self.db.execute("INSERT OR IGNORE INTO episodes VALUES(?,?,NULL)", (episode.episode_id, payload))

    def episode(self, identity):
        row = self.db.execute("SELECT payload FROM episodes WHERE id=?", (identity,)).fetchone()
        if not row:
            raise KeyError(identity)
        p = json.loads(row[0])
        return BuildEpisode(
            identity,
            p["N"],
            tuple(
                BuildAction(a["residue"], torch.tensor(a["geometry"], dtype=torch.float64), tuple(a["mask"]))
                for a in p["actions"]
            ),
            p["behavior"],
        )

    def attach(self, identity, card):
        e = self.episode(identity)
        p = GeneratedProtein(replay(e.N, e.actions))
        stored = json.loads(self.db.execute("SELECT payload FROM episodes WHERE id=?", (identity,)).fetchone()[0])
        proposal_sha = stored.get("proposal_sha256", p.geometry_sha256)
        if "proposal_ca" in stored and digest(stored["proposal_ca"]) != proposal_sha:
            raise ValueError("stored authoritative proposal corrupted")
        if (card.sequence_sha256, card.candidate_geometry_sha256) != (p.sequence_sha256, proposal_sha):
            raise ValueError("reward different proposal")
        payload = json.dumps(asdict(card), sort_keys=True, allow_nan=False)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT reward FROM episodes WHERE id=?", (identity,)).fetchone()[0]
            if old and old != payload:
                raise ValueError("immutable reward conflict")
            self.db.execute("UPDATE episodes SET reward=? WHERE id=?", (payload, identity))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def resolved_ids(self):
        return [r[0] for r in self.db.execute("SELECT id FROM episodes WHERE reward IS NOT NULL ORDER BY id")]

    def oracle_return(self, identity):
        row = self.db.execute("SELECT reward FROM episodes WHERE id=?", (identity,)).fetchone()
        if not row or row[0] is None:
            raise ValueError("pending reward UNKNOWN, never zero")
        return json.loads(row[0])

    def close(self):
        self.db.close()
