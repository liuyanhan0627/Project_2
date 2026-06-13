import hashlib
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


def validate_shared_tokenizer(big_tokenizer, small_tokenizer) -> None:
    """Fail fast when top-K token ids cannot be compared across models."""
    if big_tokenizer.get_vocab() != small_tokenizer.get_vocab():
        raise ValueError("Tokenizer mismatch: top-K distillation invalid")


def _as_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    values = logits.detach()
    if values.dim() == 2:
        values = values[0]
    return values.float()


def _topk(logits: torch.Tensor, k: int) -> Dict[str, List[Any]]:
    values = _as_1d_logits(logits)
    top_values, top_ids = torch.topk(values, k=min(k, values.shape[-1]), dim=-1)
    return {
        "ids": [int(token_id) for token_id in top_ids.detach().cpu().tolist()],
        "logits": [float(value) for value in top_values.detach().cpu().tolist()],
    }


def _gather_logits(logits: torch.Tensor, token_ids: Sequence[int]) -> List[float]:
    if not token_ids:
        return []
    values = _as_1d_logits(logits)
    ids = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long, device=values.device)
    gathered = values.gather(0, ids)
    return [float(value) for value in gathered.detach().cpu().tolist()]


def _entropy(logits: torch.Tensor) -> float:
    values = _as_1d_logits(logits)
    probs = F.softmax(values, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return float(entropy.detach().cpu().item())


class TeacherSignalCollector:
    """Collect big/small next-token distributions at ASPS trigger points.

    Records are built in memory during one sample and written by the runner
    after the final correctness label is known. The class is thread-safe enough
    for the current one-worker draft executor: the draft thread may fill in
    small-model fields while the main thread continues decoding.
    """

    def __init__(
        self,
        log_dir: str,
        top_k: int = 20,
        run_id: str = "",
        max_inline_prefix_bytes: int = 50_000,
    ):
        self.log_dir = log_dir
        self.top_k = int(top_k)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.max_inline_prefix_bytes = int(max_inline_prefix_bytes)
        self._lock = threading.Lock()
        self._prefix_hashes = set()
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def records_path(self) -> str:
        return os.path.join(self.log_dir, "trigger_records.jsonl")

    @property
    def prefix_pool_path(self) -> str:
        return os.path.join(self.log_dir, "prefix_pool.jsonl")

    def make_trigger_record(
        self,
        dataset: str,
        question_id: Any,
        trigger_step: int,
        prefix_token_ids: Sequence[int],
        big_logits: torch.Tensor,
        big_entropy: float,
    ) -> Dict[str, Any]:
        top = _topk(big_logits, self.top_k)
        return {
            "record_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "dataset": str(dataset),
            "question_id": str(question_id),
            "trigger_step": int(trigger_step),
            "prefix_token_ids": [int(token_id) for token_id in prefix_token_ids],
            "prefix_hash": None,
            "big_topk_ids": top["ids"],
            "big_topk_logits": top["logits"],
            "small_topk_ids": [],
            "small_topk_logits": [],
            "small_logits_on_big_topk": [],
            "big_logits_on_small_topk": [],
            "big_entropy": float(big_entropy),
            "small_entropy": None,
            "draft_first_token_ids": [],
            "draft_path_token_ids": [],
            "fallback_token_ids": [],
            "candidate_labels": [],
            "candidate_avg_logprobs": [],
            "candidate_length_weights": [],
            "candidate_weighted_scores": [],
            "verify_mode": None,
            "chosen_source": None,
            "accepted_label": None,
            "switch_happened": False,
            "resolution": "pending",
            "final_correct": None,
        }

    def update_small_distribution(self, record: Optional[Dict[str, Any]], small_logits: torch.Tensor) -> None:
        if record is None:
            return
        top = _topk(small_logits, self.top_k)
        small_on_big = _gather_logits(small_logits, record.get("big_topk_ids", []))
        with self._lock:
            record["small_topk_ids"] = top["ids"]
            record["small_topk_logits"] = top["logits"]
            record["small_logits_on_big_topk"] = small_on_big
            record["small_entropy"] = _entropy(small_logits)

    def update_big_logits_on_small_topk(self, record: Optional[Dict[str, Any]], big_logits: torch.Tensor) -> None:
        if record is None:
            return
        with self._lock:
            small_topk_ids = list(record.get("small_topk_ids", []))
        if not small_topk_ids:
            return
        big_on_small = _gather_logits(big_logits, small_topk_ids)
        with self._lock:
            record["big_logits_on_small_topk"] = big_on_small

    def update_draft_paths(self, record: Optional[Dict[str, Any]], drafts: Iterable[Dict[str, Any]]) -> None:
        if record is None:
            return
        paths = []
        first_ids = []
        for draft in drafts:
            token_ids = [int(token_id) for token_id in draft.get("small_token_ids", [])]
            paths.append(token_ids)
            if token_ids:
                first_ids.append(token_ids[0])
        with self._lock:
            record["draft_path_token_ids"] = paths
            record["draft_first_token_ids"] = first_ids

    def finalize_record(
        self,
        record: Optional[Dict[str, Any]],
        chosen_source: str,
        switch_happened: bool,
        resolution: str,
        fallback_token_ids: Optional[Sequence[int]] = None,
        accepted_label: Optional[str] = None,
        candidate_labels: Optional[Sequence[str]] = None,
        candidate_avg_logprobs: Optional[Sequence[float]] = None,
        candidate_length_weights: Optional[Sequence[float]] = None,
        candidate_weighted_scores: Optional[Sequence[float]] = None,
        verify_mode: Optional[str] = None,
    ) -> None:
        if record is None:
            return
        with self._lock:
            record["chosen_source"] = chosen_source
            record["switch_happened"] = bool(switch_happened)
            record["resolution"] = resolution
            record["accepted_label"] = accepted_label
            record["fallback_token_ids"] = [int(token_id) for token_id in (fallback_token_ids or [])]
            record["candidate_labels"] = list(candidate_labels or [])
            record["candidate_avg_logprobs"] = [float(value) for value in (candidate_avg_logprobs or [])]
            record["candidate_length_weights"] = [float(value) for value in (candidate_length_weights or [])]
            record["candidate_weighted_scores"] = [float(value) for value in (candidate_weighted_scores or [])]
            record["verify_mode"] = verify_mode

    def snapshot_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in records]

    def write_records(self, records: Sequence[Dict[str, Any]], final_correct: Optional[bool]) -> None:
        if not records:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with self._lock:
            serializable = [self._prepare_for_write(dict(record), final_correct) for record in records]
            with open(self.records_path, "a", encoding="utf-8") as handle:
                for record in serializable:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _prepare_for_write(self, record: Dict[str, Any], final_correct: Optional[bool]) -> Dict[str, Any]:
        record["final_correct"] = final_correct if final_correct is None else bool(final_correct)
        prefix = record.get("prefix_token_ids") or []
        prefix_json = json.dumps(prefix, separators=(",", ":"))
        if len(prefix_json.encode("utf-8")) <= self.max_inline_prefix_bytes:
            return record

        prefix_hash = hashlib.sha256(prefix_json.encode("utf-8")).hexdigest()
        record["prefix_hash"] = prefix_hash
        record["prefix_token_ids"] = []
        if prefix_hash not in self._prefix_hashes:
            self._prefix_hashes.add(prefix_hash)
            with open(self.prefix_pool_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "prefix_hash": prefix_hash,
                            "prefix_token_ids": prefix,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        return record
