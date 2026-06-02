import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GroupADecodingConfig:
    max_new_tokens: int = 600
    entropy_threshold: float = 1.5
    draft_candidates: int = 3
    max_draft_tokens: int = 128
    max_fallback_tokens: int = 128
    big_temperature: float = 0.0
    big_top_p: float = 1.0
    small_temperature: float = 0.7
    small_top_p: float = 0.9
    path_length_weight_alpha: float = 0.0
    path_length_weight_mode: str = "none"
    stop_strings: Sequence[str] = field(default_factory=lambda: ("\n\n\n",))
    use_prefix_cache_for_verify: bool = True
    punctuation: Sequence[str] = field(
        default_factory=lambda: (".", ",", "\n", "?", "!", ":", ";", ")", "]", "}")
    )


class GroupAAsyncDecoder:
    """CNTP-style Group A decoder.

    The target model keeps decoding a fallback punctuation span while the draft
    model proposes short punctuation-bounded paths. If drafts return before the
    one-span rollback window closes, the target model scores every candidate in
    one batched verification pass and switches by the configured path score.
    """

    def __init__(self, big_model, big_tokenizer, small_model, small_tokenizer, config: GroupADecodingConfig):
        self.big_model = big_model
        self.big_tokenizer = big_tokenizer
        self.small_model = small_model
        self.small_tokenizer = small_tokenizer
        self.config = config
        self.big_device = next(big_model.parameters()).device
        self.small_device = next(small_model.parameters()).device
        self.big_punctuation_ids = self._punctuation_ids(big_tokenizer)
        self.small_punctuation_ids = self._punctuation_ids(small_tokenizer)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _punctuation_ids(self, tokenizer) -> set:
        ids = set()
        for mark in self.config.punctuation:
            encoded = tokenizer.encode(mark, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return ids

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _sampling_scores(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        scores = logits.float()
        if temperature and temperature > 0:
            scores = scores / temperature
        return self._apply_top_p(scores, top_p)

    def _select_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        scores = self._sampling_scores(logits, temperature, top_p)
        if temperature and temperature > 0:
            probs = F.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.argmax(scores, dim=-1)

    def _entropy(self, logits: torch.Tensor) -> float:
        scores = self._sampling_scores(logits, self.config.big_temperature, self.config.big_top_p)
        probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return float(entropy.item())

    def _clone_cache(self, past_key_values):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, tuple):
            return tuple(self._clone_cache(layer) for layer in past_key_values)
        if isinstance(past_key_values, list):
            return [self._clone_cache(layer) for layer in past_key_values]
        if torch.is_tensor(past_key_values):
            return past_key_values.clone()
        return copy.deepcopy(past_key_values)

    def _repeat_cache(self, past_key_values, repeats: int):
        cache = self._clone_cache(past_key_values)
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(repeats)
            return cache
        if isinstance(cache, tuple):
            return tuple(self._repeat_cache(layer, repeats) for layer in cache)
        if isinstance(cache, list):
            return [self._repeat_cache(layer, repeats) for layer in cache]
        if torch.is_tensor(cache):
            return cache.repeat_interleave(repeats, dim=0)
        return cache

    def _decode_text(self, tokenizer, token_ids: Sequence[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _encode_text(self, tokenizer, text: str, device) -> torch.LongTensor:
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    @torch.no_grad()
    def _draft_paths(self, trigger_text: str) -> List[Dict]:
        candidate_count = max(1, self.config.draft_candidates)
        base = self.small_tokenizer(trigger_text, return_tensors="pt").to(self.small_device)
        input_ids = base.input_ids.repeat(candidate_count, 1)
        attention_mask = base.attention_mask.repeat(candidate_count, 1)
        outputs = self.small_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        draft_tokens = [[] for _ in range(candidate_count)]
        active = torch.ones(candidate_count, dtype=torch.bool, device=self.small_device)
        eos_id = self.small_tokenizer.eos_token_id

        for _step in range(self.config.max_draft_tokens):
            next_token = self._select_token(next_logits, self.config.small_temperature, self.config.small_top_p)
            step_input = next_token.clone()
            for row, token_id in enumerate(next_token.tolist()):
                if not bool(active[row].item()):
                    step_input[row] = eos_id
                    continue
                draft_tokens[row].append(int(token_id))
                if token_id in self.small_punctuation_ids or token_id == eos_id:
                    active[row] = False
            if not bool(active.any().item()):
                break

            step_input = step_input[:, None]
            step_attention = torch.ones(
                (candidate_count, input_ids.shape[1] + 1),
                dtype=torch.long,
                device=self.small_device,
            )
            outputs = self.small_model(
                input_ids=step_input,
                attention_mask=step_attention,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            input_ids = torch.cat([input_ids, step_input], dim=-1)

        paths = []
        for tokens in draft_tokens:
            text = self._decode_text(self.small_tokenizer, tokens)
            paths.append({"text": text, "small_token_ids": tokens})
        return paths

    @torch.no_grad()
    def _advance_big_one_token(
        self,
        input_ids: torch.LongTensor,
        past_key_values,
        next_logits: torch.Tensor,
    ) -> Tuple[torch.LongTensor, object, torch.Tensor, int]:
        next_token = self._select_token(next_logits, self.config.big_temperature, self.config.big_top_p)
        token_id = int(next_token.item())
        step_input = next_token[:, None]
        attention_mask = torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long, device=self.big_device)
        outputs = self.big_model(
            input_ids=step_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        input_ids = torch.cat([input_ids, step_input], dim=-1)
        return input_ids, outputs.past_key_values, outputs.logits[:, -1, :], token_id

    @torch.no_grad()
    def _fallback_span(
        self,
        input_ids: torch.LongTensor,
        past_key_values,
        next_logits: torch.Tensor,
    ) -> Tuple[torch.LongTensor, object, torch.Tensor, List[int]]:
        fallback_tokens = []
        for _ in range(self.config.max_fallback_tokens):
            input_ids, past_key_values, next_logits, token_id = self._advance_big_one_token(
                input_ids, past_key_values, next_logits
            )
            fallback_tokens.append(token_id)
            if token_id in self.big_punctuation_ids or token_id == self.big_tokenizer.eos_token_id:
                break
        return input_ids, past_key_values, next_logits, fallback_tokens

    def _nonempty_candidate_ids(self, text: str) -> Optional[List[int]]:
        token_ids = self.big_tokenizer.encode(text, add_special_tokens=False)
        token_ids = [int(t) for t in token_ids if t != self.big_tokenizer.pad_token_id]
        return token_ids or None

    @torch.no_grad()
    def _score_candidates_full_batch(
        self, trigger_prefix_ids: torch.LongTensor, candidate_token_ids: List[List[int]]
    ) -> List[float]:
        prefix = trigger_prefix_ids[0].tolist()
        sequences = [prefix + ids for ids in candidate_token_ids]
        max_len = max(len(seq) for seq in sequences)
        pad_id = self.big_tokenizer.pad_token_id
        batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=self.big_device)
        attention_mask = torch.zeros_like(batch)
        for row, seq in enumerate(sequences):
            batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.big_device)
            attention_mask[row, : len(seq)] = 1

        outputs = self.big_model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

        prefix_len = len(prefix)
        scores = []
        for row, ids in enumerate(candidate_token_ids):
            total = 0.0
            for offset, token_id in enumerate(ids):
                pred_pos = prefix_len + offset - 1
                total += float(log_probs[row, pred_pos, token_id].item())
            scores.append(total / max(1, len(ids)))
        return scores

    @torch.no_grad()
    def _score_candidates_with_prefix_cache(
        self,
        trigger_prefix_ids: torch.LongTensor,
        prefix_past,
        prefix_next_logits: torch.Tensor,
        candidate_token_ids: List[List[int]],
    ) -> List[float]:
        batch_size = len(candidate_token_ids)
        lengths = [len(ids) for ids in candidate_token_ids]
        first_ids = torch.tensor([ids[0] for ids in candidate_token_ids], device=self.big_device)
        first_log_probs = F.log_softmax(prefix_next_logits.float(), dim=-1)[0, first_ids]
        totals = first_log_probs.float()

        max_prev_len = max(length - 1 for length in lengths)
        if max_prev_len > 0:
            pad_id = self.big_tokenizer.pad_token_id
            prev_tokens = torch.full((batch_size, max_prev_len), pad_id, dtype=torch.long, device=self.big_device)
            prev_mask = torch.zeros_like(prev_tokens)
            for row, ids in enumerate(candidate_token_ids):
                prev = ids[:-1]
                if prev:
                    prev_tokens[row, : len(prev)] = torch.tensor(prev, dtype=torch.long, device=self.big_device)
                    prev_mask[row, : len(prev)] = 1

            expanded_cache = self._repeat_cache(prefix_past, batch_size)
            prefix_len = trigger_prefix_ids.shape[1]
            attention_mask = torch.cat(
                [
                    torch.ones((batch_size, prefix_len), dtype=torch.long, device=self.big_device),
                    prev_mask,
                ],
                dim=-1,
            )
            outputs = self.big_model(
                input_ids=prev_tokens,
                attention_mask=attention_mask,
                past_key_values=expanded_cache,
                use_cache=False,
            )
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
            for row, ids in enumerate(candidate_token_ids):
                for pos, token_id in enumerate(ids[1:]):
                    totals[row] += log_probs[row, pos, token_id]

        return (totals / torch.tensor(lengths, dtype=torch.float, device=self.big_device)).tolist()

    @torch.no_grad()
    def _score_candidates(
        self,
        trigger_prefix_ids: torch.LongTensor,
        prefix_past,
        prefix_next_logits: torch.Tensor,
        candidate_token_ids: List[List[int]],
    ) -> Tuple[List[float], str]:
        if self.config.use_prefix_cache_for_verify:
            try:
                return (
                    self._score_candidates_with_prefix_cache(
                        trigger_prefix_ids, prefix_past, prefix_next_logits, candidate_token_ids
                    ),
                    "prefix_cache",
                )
            except Exception as exc:
                print(f"[GroupA] prefix-cache verification failed; falling back to full-batch scoring: {exc}")
        return self._score_candidates_full_batch(trigger_prefix_ids, candidate_token_ids), "full_batch"

    def _candidate_length_weights(self, candidate_token_ids: List[List[int]]) -> List[float]:
        if self.config.path_length_weight_alpha <= 0 or self.config.path_length_weight_mode == "none":
            return [0.0 for _ in candidate_token_ids]

        lengths = [max(1, len(ids)) for ids in candidate_token_ids]
        max_len = max(lengths)
        min_len = min(lengths)
        mode = self.config.path_length_weight_mode

        if mode == "shorter":
            return [min_len / length for length in lengths]
        if mode == "log_longer":
            denom = max(1e-6, math.log1p(max_len))
            return [math.log1p(length) / denom for length in lengths]
        return [length / max_len for length in lengths]

    def _should_stop_text(self, text: str) -> bool:
        return any(stop and stop in text for stop in self.config.stop_strings)

    @torch.no_grad()
    def generate(self, prompt_text: str) -> Dict:
        encoded = self.big_tokenizer(prompt_text, return_tensors="pt").to(self.big_device)
        input_ids = encoded.input_ids
        prompt_len = input_ids.shape[1]
        outputs = self.big_model(input_ids=input_ids, attention_mask=encoded.attention_mask, use_cache=True)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        metrics = {
            "entropy_triggers": 0,
            "draft_calls": 0,
            "small_draft_wins": 0,
            "fallback_wins": 0,
            "switches": 0,
            "late_draft_drops": 0,
            "wasted_big_tokens": 0,
            "verify_calls": 0,
            "verify_modes": {},
            "candidate_ppl": [],
            "candidate_base_scores": [],
            "candidate_length_weights": [],
            "candidate_weighted_scores": [],
            "length_weight_overrides": 0,
            "accepted_tokens_per_verify": [],
            "fallback_path_lengths": [],
            "draft_path_lengths": [],
            "wall_time": 0.0,
        }
        start_time = time.time()

        while input_ids.shape[1] - prompt_len < self.config.max_new_tokens:
            generated_text = self.big_tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
            if self._should_stop_text(generated_text):
                break
            entropy = self._entropy(next_logits)
            if entropy <= self.config.entropy_threshold:
                input_ids, past, next_logits, token_id = self._advance_big_one_token(input_ids, past, next_logits)
                if token_id == self.big_tokenizer.eos_token_id:
                    break
                continue

            metrics["entropy_triggers"] += 1
            metrics["draft_calls"] += 1
            trigger_prefix_ids = input_ids.clone()
            trigger_prefix_text = self.big_tokenizer.decode(trigger_prefix_ids[0], skip_special_tokens=True)
            prefix_past = self._clone_cache(past)
            prefix_next_logits = next_logits.clone()
            draft_future = self.executor.submit(self._draft_paths, trigger_prefix_text)

            input_ids, past, next_logits, fallback_tokens = self._fallback_span(input_ids, past, next_logits)
            metrics["fallback_path_lengths"].append(len(fallback_tokens))
            if not draft_future.done():
                metrics["late_draft_drops"] += 1
                draft_future.cancel()
                continue

            drafts = draft_future.result()
            candidates = []
            labels = []
            for idx, draft in enumerate(drafts):
                candidate_ids = self._nonempty_candidate_ids(draft["text"])
                if candidate_ids is None:
                    continue
                candidates.append(candidate_ids)
                labels.append(f"small_{idx}")
                metrics["draft_path_lengths"].append(len(candidate_ids))
            if fallback_tokens:
                candidates.append(fallback_tokens)
                labels.append("fallback")
            if not candidates:
                continue

            avg_logprobs, verify_mode = self._score_candidates(
                trigger_prefix_ids, prefix_past, prefix_next_logits, candidates
            )
            metrics["verify_calls"] += 1
            metrics["verify_modes"][verify_mode] = metrics["verify_modes"].get(verify_mode, 0) + 1
            ppls = [math.exp(-score) for score in avg_logprobs]
            metrics["candidate_ppl"].append(dict(zip(labels, ppls)))
            length_weights = self._candidate_length_weights(candidates)
            weighted_scores = [
                score + self.config.path_length_weight_alpha * weight
                for score, weight in zip(avg_logprobs, length_weights)
            ]
            best_base_idx = max(range(len(avg_logprobs)), key=lambda i: avg_logprobs[i])
            best_idx = max(range(len(weighted_scores)), key=lambda i: weighted_scores[i])
            best_label = labels[best_idx]
            metrics["accepted_tokens_per_verify"].append(len(candidates[best_idx]))
            if self.config.path_length_weight_alpha > 0 and self.config.path_length_weight_mode != "none":
                metrics["candidate_base_scores"].append(dict(zip(labels, avg_logprobs)))
                metrics["candidate_length_weights"].append(dict(zip(labels, length_weights)))
                metrics["candidate_weighted_scores"].append(dict(zip(labels, weighted_scores)))
                if best_idx != best_base_idx:
                    metrics["length_weight_overrides"] += 1

            if best_label == "fallback":
                metrics["fallback_wins"] += 1
                continue

            metrics["small_draft_wins"] += 1
            metrics["switches"] += 1
            metrics["wasted_big_tokens"] += len(fallback_tokens)
            best_tokens = torch.tensor([candidates[best_idx]], dtype=torch.long, device=self.big_device)
            input_ids = torch.cat([trigger_prefix_ids, best_tokens], dim=-1)
            attention_mask = torch.ones_like(input_ids)
            outputs = self.big_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        metrics["wall_time"] = time.time() - start_time
        gen_ids = input_ids[0, prompt_len:]
        text = self.big_tokenizer.decode(gen_ids, skip_special_tokens=True)
        tokens = [
            tok.replace("▁", " ").replace("<0x0A>", "\n")
            for tok in self.big_tokenizer.convert_ids_to_tokens(gen_ids)
        ]
        return {
            "text": text,
            "token_ids": gen_ids.detach().cpu().tolist(),
            "tokens": tokens,
            "metrics": metrics,
        }
