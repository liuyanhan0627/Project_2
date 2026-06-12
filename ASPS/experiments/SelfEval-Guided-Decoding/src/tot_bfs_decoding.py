import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig


@dataclass
class ToTBFSConfig:
    max_depth: int = 3
    thought_branching: int = 3
    beam_width: int = 2
    n_evaluate_sample: int = 1
    thought_temperature: float = 0.7
    thought_top_p: float = 0.95
    value_temperature: float = 0.0
    value_top_p: float = 1.0
    final_temperature: float = 0.0
    final_top_p: float = 1.0
    max_thought_tokens: int = 96
    max_value_tokens: int = 32
    max_final_tokens: int = 256
    stop_strings: Sequence[str] = ("\n\n\n",)


@dataclass
class ToTState:
    thoughts: List[str]
    score: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(f"{idx + 1}. {thought}" for idx, thought in enumerate(self.thoughts))


class ToTBFSDecoder:
    """Classic Tree-of-Thoughts BFS/beam decoder using a local HF causal LM."""

    def __init__(self, model, tokenizer, config: ToTBFSConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def _generation_config(self, max_new_tokens: int, temperature: float, top_p: float, n: int) -> GenerationConfig:
        do_sample = bool(temperature and temperature > 0)
        if n > 1:
            do_sample = True
            temperature = max(float(temperature), 1e-6)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.stop_strings:
            kwargs["stop_strings"] = list(self.config.stop_strings)
        return GenerationConfig(**kwargs)

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        n: int = 1,
    ) -> Tuple[List[str], Dict[str, int]]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = self._generation_config(max_new_tokens, temperature, top_p, n)
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )
        input_len = inputs.input_ids.shape[1]
        texts = []
        output_tokens = 0
        for row in outputs:
            gen_ids = row[input_len:]
            output_tokens += int(gen_ids.numel())
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return texts, {
            "input_tokens": int(inputs.input_ids.numel()) * n,
            "output_tokens": output_tokens,
        }

    def _question_prompt(self, dt_name: str, question: str) -> str:
        if dt_name == "gsm8k":
            return (
                "Solve the arithmetic word problem. Build the solution as a sequence of concise reasoning steps.\n\n"
                f"Problem:\n{question.strip()}"
            )
        if dt_name == "strategyqa":
            return (
                "Answer the yes/no question by building a sequence of concise factual reasoning steps.\n\n"
                f"Question:\n{question.strip()}"
            )
        if dt_name == "math":
            return (
                "Solve the math problem by building a sequence of concise reasoning steps.\n\n"
                f"Problem:\n{question.strip()}"
            )
        return f"Question:\n{question.strip()}"

    def _thought_prompt(self, dt_name: str, question: str, state: ToTState) -> str:
        current = state.text or "(no previous thoughts)"
        return (
            f"{self._question_prompt(dt_name, question)}\n\n"
            f"Current reasoning path:\n{current}\n\n"
            "Generate exactly one next useful thought for this path. "
            "Keep it short, concrete, and do not include a final answer unless the answer follows immediately.\n"
            "Next thought:"
        )

    def _value_prompt(self, dt_name: str, question: str, state: ToTState) -> str:
        answer_hint = {
            "gsm8k": "a correct arithmetic solution",
            "strategyqa": "a correct yes/no answer",
            "math": "a correct boxed mathematical answer",
        }.get(dt_name, "a correct answer")
        return (
            f"{self._question_prompt(dt_name, question)}\n\n"
            f"Candidate reasoning path:\n{state.text or '(empty)'}\n\n"
            f"Score how promising this path is for reaching {answer_hint}. "
            "Use an integer from 0 to 10, where 0 is impossible, 5 is uncertain, and 10 is certainly correct.\n"
            "Return exactly one line in this format: Score: <integer>"
        )

    def _format_path_for_python_comment(self, state: ToTState) -> str:
        lines = state.text.splitlines() or ["No reliable reasoning path was selected."]
        return "\n".join(f"# {line}" for line in lines)

    def _final_prompt(self, dt_name: str, full_prompt: str, question: str, state: ToTState) -> str:
        if dt_name == "gsm8k":
            return (
                f"{full_prompt.rstrip()}\n"
                "# Tree-of-Thoughts selected reasoning path:\n"
                f"{self._format_path_for_python_comment(state)}\n"
                "# Now write the final answer as Python code only.\n"
                "# Define def solution(): and return the numeric result.\n\n"
            )
        if dt_name == "strategyqa":
            return (
                f"{full_prompt.rstrip()}\n"
                "Tree-of-Thoughts selected reasoning path:\n"
                f"{state.text or 'No reliable reasoning path was selected.'}\n\n"
                "Now write the final reasoning concisely. End exactly with "
                '"So the answer is yes." or "So the answer is no."\n'
            )
        if dt_name == "math":
            return (
                f"{full_prompt.rstrip()}\n"
                "Tree-of-Thoughts selected reasoning path:\n"
                f"{state.text or 'No reliable reasoning path was selected.'}\n\n"
                "Now write a concise final solution. Put the final answer within \\boxed{}.\n"
            )
        return (
            f"{full_prompt.rstrip()}\n"
            f"Tree-of-Thoughts selected reasoning path:\n{state.text}\n\n"
            "Now write the final answer.\n"
        )

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _clean_thought(self, text: str) -> Optional[str]:
        text = self._strip_code_fence(text)
        text = re.split(r"\n\s*\n|(?:^|\n)\s*(?:Score|Evaluation|Next thought)\s*:", text, maxsplit=1)[0]
        text = text.strip().strip("-* ")
        if not text:
            return None
        return re.sub(r"\s+", " ", text)[:800]

    def _clean_final(self, dt_name: str, text: str) -> str:
        text = self._strip_code_fence(text)
        if dt_name == "gsm8k":
            marker = "def solution"
            pos = text.find(marker)
            if pos != -1:
                text = text[pos:]
                lines = text.splitlines()
                kept = []
                started = False
                for line in lines:
                    if line.strip():
                        started = True
                    if started and kept and line and not line.startswith((" ", "\t", "#")):
                        break
                    kept.append(line)
                text = "\n".join(kept).strip()
            return text
        if dt_name == "strategyqa":
            match = re.search(r"So the answer is\s+(yes|no)\.", text, flags=re.I)
            if match:
                return text[: match.end()].strip()
        return text.strip()

    def _parse_score(self, text: str) -> float:
        match = re.search(r"Score\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.I)
        if not match:
            match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        score = float(match.group(1) if match.lastindex else match.group(0))
        return max(0.0, min(10.0, score))

    def _score_state(self, dt_name: str, question: str, state: ToTState, metrics: Dict) -> float:
        scores = []
        prompt = self._value_prompt(dt_name, question, state)
        for _ in range(max(1, self.config.n_evaluate_sample)):
            texts, usage = self._generate(
                prompt,
                self.config.max_value_tokens,
                self.config.value_temperature,
                self.config.value_top_p,
                n=1,
            )
            metrics["evaluate_calls"] += 1
            metrics["big_input_tokens"] += usage["input_tokens"]
            metrics["big_output_tokens"] += usage["output_tokens"]
            scores.append(self._parse_score(texts[0]))
        score = sum(scores) / max(1, len(scores))
        return float(score)

    def generate(self, full_prompt: str, question: str, dt_name: str) -> Dict:
        start_time = time.time()
        metrics = {
            "tot_depth": self.config.max_depth,
            "thought_branching": self.config.thought_branching,
            "beam_width": self.config.beam_width,
            "n_evaluate_sample": self.config.n_evaluate_sample,
            "generate_calls": 0,
            "evaluate_calls": 0,
            "final_calls": 0,
            "generated_thoughts": 0,
            "evaluated_states": 0,
            "selected_state_counts": [],
            "selected_scores_by_depth": [],
            "avg_state_score": None,
            "best_state_score": None,
            "big_input_tokens": 0,
            "big_output_tokens": 0,
            "wall_time": 0.0,
        }
        all_scores: List[float] = []
        states = [ToTState(thoughts=[])]

        for _depth in range(max(1, self.config.max_depth)):
            candidates: List[ToTState] = []
            seen = set()
            for state in states:
                prompt = self._thought_prompt(dt_name, question, state)
                thoughts, usage = self._generate(
                    prompt,
                    self.config.max_thought_tokens,
                    self.config.thought_temperature,
                    self.config.thought_top_p,
                    n=max(1, self.config.thought_branching),
                )
                metrics["generate_calls"] += 1
                metrics["big_input_tokens"] += usage["input_tokens"]
                metrics["big_output_tokens"] += usage["output_tokens"]
                for thought_text in thoughts:
                    thought = self._clean_thought(thought_text)
                    if thought is None:
                        continue
                    key = (tuple(state.thoughts), thought.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = ToTState(thoughts=state.thoughts + [thought])
                    candidate.score = self._score_state(dt_name, question, candidate, metrics)
                    candidates.append(candidate)
                    all_scores.append(candidate.score)

            metrics["generated_thoughts"] += len(candidates)
            metrics["evaluated_states"] += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda item: item.score, reverse=True)
            states = candidates[: max(1, self.config.beam_width)]
            metrics["selected_state_counts"].append(len(states))
            metrics["selected_scores_by_depth"].append([state.score for state in states])

        best_state = max(states, key=lambda item: item.score) if states else ToTState(thoughts=[])
        final_prompt = self._final_prompt(dt_name, full_prompt, question, best_state)
        final_texts, usage = self._generate(
            final_prompt,
            self.config.max_final_tokens,
            self.config.final_temperature,
            self.config.final_top_p,
            n=1,
        )
        metrics["final_calls"] += 1
        metrics["big_input_tokens"] += usage["input_tokens"]
        metrics["big_output_tokens"] += usage["output_tokens"]
        metrics["avg_state_score"] = sum(all_scores) / len(all_scores) if all_scores else None
        metrics["best_state_score"] = best_state.score if best_state.thoughts else None
        metrics["wall_time"] = time.time() - start_time

        text = self._clean_final(dt_name, final_texts[0] if final_texts else "")
        tokens = self.tokenizer.tokenize(text)
        return {
            "text": text,
            "tokens": tokens,
            "tot_path": best_state.text,
            "metrics": metrics,
        }
