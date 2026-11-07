"""Sentence pool for the typing tasks, the vowel rule, and scoring of typed answers.

Run it to print the pool's length and vowel stats and write items.json:

    uv run python tasks.py
"""

import json
import statistics
from pathlib import Path

VOWELS = frozenset("aeiou")
CAPITAL_VOWELS = frozenset("AEIOU")

# Candidates are filtered by sentence_ok(); POOL keeps the first 40 that pass.
CANDIDATES = [
    "the little dog ran across the park to find a ball",
    "we made fresh bread and soup for our friends on sunday",
    "a warm cup of tea is nice on a cold rainy morning",
    "my sister likes to read books under the old oak tree",
    "the children played outside until the sun went down",
    "she painted a blue boat on the side of the garden wall",
    "our team went for a long walk along the quiet beach",
    "he found an old map hidden inside a wooden box",
    "the baker sells warm cakes and sweet buns every day",
    "a small bird sang a happy song outside my window",
    "they planted tomatoes and beans in the back garden",
    "the kind teacher gave each student a bright gold star",
    "we watched the stars from the roof on a clear night",
    "my uncle cooked a big pot of pasta for the whole family",
    "the cat slept on a soft blanket next to the fire",
    "a gentle breeze moved the leaves on the apple tree",
    "the old train rolled slowly over the stone bridge",
    "she wrote a long letter to her friend in another town",
    "we ate ice cream on the pier and watched the boats",
    "the farmer opened the gate and the sheep ran out",
    "a bright orange kite flew high above the open field",
    "he likes to ride his bike to the lake on warm days",
    "the museum was full of strange and beautiful objects",
    "our neighbour baked a pie and shared it with us",
    "a tiny mouse ate the cheese and ran back to its hole",
    "the river was calm and the water looked very clear",
    "we sang songs around the campfire under the moon",
    "the young girl found a pretty shell on the sand",
    "my brother built a robot out of paper and old boxes",
    "the library opens early and closes late on fridays",
    "a family of ducks swam across the pond in a line",
    "she bought apples and grapes at the busy market",
    "the dog waited by the door for his owner to come home",
    "we took a boat to the island and had a picnic there",
    "the snow covered the hills and the roads were quiet",
    "an old man fed bread to the pigeons in the square",
    "the band played music while people danced in the hall",
    "he opened the window to let some fresh air inside",
    "a red balloon floated up into the blue evening sky",
    "the twins made a cake for their mother on her birthday",
    "our class visited a farm and fed the baby goats",
    "the moon was huge and orange above the dark sea",
    "she keeps her seeds in a jar on the kitchen shelf",
    "the old clock in the hall ticks loudly every night",
    "we found a quiet cafe and ordered two hot drinks",
    "a friendly horse ate an apple from my open hand",
    "the rain stopped and a rainbow appeared over the town",
    "my aunt taught me how to bake bread on a rainy day",
    "the puppy chased its tail around the living room",
    "a boat with white sails moved across the calm bay",
    "the kids built a snowman with a carrot for a nose",
    "he plays the piano every evening after his dinner",
    "the garden was full of roses and tall yellow flowers",
    "we climbed the hill to see the sun rise over the sea",
    "an owl hooted softly in the woods behind our house",
    "the shop on the corner sells fresh fruit and flowers",
    "our grandmother tells the best stories before bed",
    "a little frog jumped into the pond with a big splash",
    "the bus was late so we walked home through the park",
    "she gave the hungry cat a bowl of warm milk",
    "the pilot waved as the plane rolled onto the runway",
    "a curious fox looked at us from behind the old fence",
    "we cleaned the house and opened all of the windows",
    "the bakery smells of cinnamon every single morning",
    "my friend and i went fishing at the lake on saturday",
]


def capitalise_vowels(sentence: str) -> str:
    """Return the sentence with every a, e, i, o and u capitalised (y stays lowercase)."""
    return "".join(c.upper() if c in VOWELS else c for c in sentence)


def vowel_count(sentence: str) -> int:
    return sum(c in VOWELS for c in sentence)


def sentence_ok(sentence: str) -> bool:
    """8-11 words, 40-55 characters, lowercase letters and single spaces only, 13-18 vowels."""
    words = sentence.split(" ")
    return (
        8 <= len(words) <= 11
        and all(words)
        and 40 <= len(sentence) <= 55
        and all("a" <= c <= "z" or c == " " for c in sentence)
        and 13 <= vowel_count(sentence) <= 18
    )


POOL = [s for s in dict.fromkeys(CANDIDATES) if sentence_ok(s)][:40]


def align(typed: str, expected: str) -> tuple[int, list[tuple[str, str | None, str | None]]]:
    """Levenshtein distance between typed and expected, plus the edits.

    Each edit is ("sub", typed_char, expected_char), ("ins", typed_char, None) for an extra
    character, or ("del", None, expected_char) for a missing one.
    """
    n, m = len(typed), len(expected)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = typed[i - 1] == expected[j - 1]
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + (not same))
    edits = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (typed[i - 1] != expected[j - 1]):
            if typed[i - 1] != expected[j - 1]:
                edits.append(("sub", typed[i - 1], expected[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            edits.append(("ins", typed[i - 1], None))
            i -= 1
        else:
            edits.append(("del", None, expected[j - 1]))
            j -= 1
    return dist[n][m], edits[::-1]


def score(typed: str, expected: str) -> dict:
    """Edit distance, accuracy and error counts for one typed answer.

    missed_capitals: a lowercase letter where its capital was expected.
    wrong_capitals: a capital where the lowercase letter was expected (a consonant, under the rule).
    other_errors: every other wrong, extra or missing character.
    """
    distance, edits = align(typed, expected)
    missed = wrong = other = 0
    for kind, t, e in edits:
        if kind == "sub" and t.lower() == e.lower():
            if e.isupper():
                missed += 1
            else:
                wrong += 1
        else:
            other += 1
    return {
        "distance": distance,
        "accuracy": max(0.0, 1 - distance / max(len(expected), 1)),
        "missed_capitals": missed,
        "wrong_capitals": wrong,
        "other_errors": other,
    }


def items() -> list[dict]:
    return [{"id": i, "sentence": s, "rule_expected": capitalise_vowels(s)} for i, s in enumerate(POOL)]


def main() -> None:
    lengths = [len(s) for s in POOL]
    vowels = [vowel_count(s) for s in POOL]
    words = [len(s.split()) for s in POOL]
    print(f"{len(POOL)} sentences kept from {len(CANDIDATES)} candidates")
    print(f"characters: {min(lengths)}-{max(lengths)}, mean {statistics.mean(lengths):.1f}")
    print(f"vowels:     {min(vowels)}-{max(vowels)}, mean {statistics.mean(vowels):.1f}")
    print(f"words:      {min(words)}-{max(words)}, mean {statistics.mean(words):.1f}")
    out = Path(__file__).with_name("items.json")
    out.write_text(json.dumps(items(), indent=2) + "\n")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
