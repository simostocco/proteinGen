"""Deterministic amino-acid vocabulary for sequence-only protein generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CANONICAL_AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS: tuple[str, ...] = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
DEFAULT_TOKENS: tuple[str, ...] = SPECIAL_TOKENS + CANONICAL_AMINO_ACIDS


@dataclass(frozen=True)
class ProteinVocabulary:
    """Stable residue vocabulary with explicit special tokens."""

    tokens: tuple[str, ...] = DEFAULT_TOKENS

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary tokens must be unique")
        for token in SPECIAL_TOKENS:
            if token not in self.tokens:
                raise ValueError(f"Vocabulary is missing required token {token}")
        for residue in CANONICAL_AMINO_ACIDS:
            if residue not in self.tokens:
                raise ValueError(f"Vocabulary is missing canonical residue {residue}")

    @property
    def token_to_id(self) -> dict[str, int]:
        """Return token-to-id mapping."""
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def id_to_token(self) -> dict[int, str]:
        """Return id-to-token mapping."""
        return {index: token for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def canonical_ids(self) -> list[int]:
        """Return ids for the 20 biological output residues."""
        mapping = self.token_to_id
        return [mapping[token] for token in CANONICAL_AMINO_ACIDS]

    @property
    def special_ids(self) -> set[int]:
        """Return ids that are not allowed in biological generations."""
        mapping = self.token_to_id
        return {mapping[token] for token in SPECIAL_TOKENS}

    def encode(self, sequence: str, *, allow_unknown: bool = False) -> list[int]:
        """Encode a biological amino-acid sequence without adding special tokens."""
        normalized = normalize_sequence(sequence)
        mapping = self.token_to_id
        encoded: list[int] = []
        for residue in normalized:
            if residue in CANONICAL_AMINO_ACIDS:
                encoded.append(mapping[residue])
            elif allow_unknown:
                encoded.append(self.unk_id)
            else:
                raise ValueError(f"Invalid noncanonical residue {residue!r}")
        return encoded

    def decode(self, token_ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        """Decode token ids into a sequence string."""
        reverse = self.id_to_token
        residues: list[str] = []
        for token_id in token_ids:
            token = reverse[int(token_id)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            residues.append(token)
        return "".join(residues)

    def validate_canonical(self, sequence: str) -> None:
        """Raise if a sequence contains noncanonical symbols."""
        self.encode(sequence, allow_unknown=False)

    def to_json(self, path: str | Path) -> None:
        """Serialize the vocabulary to JSON."""
        Path(path).write_text(json.dumps({"tokens": list(self.tokens)}, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> ProteinVocabulary:
        """Load a vocabulary from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(tokens=tuple(data["tokens"]))


def normalize_sequence(sequence: str) -> str:
    """Normalize a biological sequence and reject whitespace."""
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a string")
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("sequence is empty")
    if any(char.isspace() for char in normalized):
        raise ValueError("sequence contains whitespace")
    return normalized
