#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


ADJECTIVES = [
    "amber", "brisk", "calm", "distant", "ember", "frost", "golden", "hidden",
    "ivory", "jade", "kind", "lunar", "misty", "navy", "opal", "quiet",
    "rapid", "silver", "tidal", "violet",
]

NOUNS = [
    "anchor", "beacon", "canyon", "delta", "engine", "forest", "garden", "harbor",
    "island", "journal", "kernel", "lantern", "meadow", "needle", "orbit", "prairie",
    "quartz", "river", "signal", "tower",
]

NOISE_SENTENCES = [
    "The archive records routine observations about weather, logistics, and maintenance.",
    "Several unrelated notes describe workshop schedules and inventory updates.",
    "A reviewer added a paragraph about historical context that is not relevant to the key.",
    "The document continues with neutral filler text intended to lengthen the context.",
    "Another section summarizes ordinary project status messages and meeting reminders.",
    "The following sentence is background material and does not contain the requested value.",
    "Participants discussed travel plans, storage labels, and generic office procedures.",
    "The report includes repeated narrative details to simulate a long-context haystack.",
]


def make_value(index):
    adjective = ADJECTIVES[index % len(ADJECTIVES)]
    noun = NOUNS[(index * 7) % len(NOUNS)]
    number = 731000 + index * 37
    return f"{adjective}-{noun}-{number}"


def build_context(index, target_words, rng):
    key = f"ruler-key-{index:03d}"
    value = make_value(index)
    needle = f"Important retrieval record: the secret value for {key} is {value}."
    sentences = []
    word_count = 0
    insert_after = int(target_words * (0.15 + 0.7 * ((index * 17) % 100) / 99.0))
    noise_index = 0
    inserted = False

    while word_count < target_words:
        if not inserted and word_count >= insert_after:
            sentences.append(needle)
            word_count += len(needle.split())
            inserted = True
            continue
        base = NOISE_SENTENCES[noise_index % len(NOISE_SENTENCES)]
        tag = rng.choice(ADJECTIVES) + "-" + rng.choice(NOUNS)
        sentence = f"{base} Section marker {index:03d}-{noise_index:04d}-{tag}."
        sentences.append(sentence)
        word_count += len(sentence.split())
        noise_index += 1

    if not inserted:
        sentences.append(needle)
    return "\n".join(sentences), key, value


def make_example(index, target_words, seed):
    rng = random.Random(seed + index * 997)
    context, key, value = build_context(index, target_words, rng)
    prompt = (
        "You are given a long document. Retrieve the exact secret value requested by the question.\n"
        "Answer with only the secret value and no explanation.\n\n"
        "<document>\n"
        f"{context}\n"
        "</document>\n\n"
        f"Question: What is the secret value for {key}?\n"
        "Answer:"
    )
    return {
        "index": index,
        "input": prompt,
        "outputs": [value],
        "length": len(prompt.split()),
        "metadata": {
            "task": "ruler_niah",
            "key": key,
            "needle": value,
            "target_words": target_words,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a deterministic small RULER/NIAH JSONL file.")
    parser.add_argument(
        "--output",
        default="ASPS/experiments/SelfEval-Guided-Decoding/data/ruler/ruler_niah_words_2k_small.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--count", default=20, type=int, help="Number of examples to generate.")
    parser.add_argument("--target-words", default=1800, type=int, help="Approximate context words per example.")
    parser.add_argument("--seed", default=20260608, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.count):
            handle.write(json.dumps(make_example(index, args.target_words, args.seed), ensure_ascii=False) + "\n")
    print(f"Wrote {args.count} RULER/NIAH examples to {output}")


if __name__ == "__main__":
    main()
