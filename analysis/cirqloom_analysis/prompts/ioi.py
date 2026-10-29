"""IOI prompt suite (M3 step 4): Wang et al. (2022) templates, with named token positions.

Templates, names, places and objects are copied from the official code,
redwoodresearch/Easy-Transformer, easy_transformer/ioi_dataset.py at commit e168301
(BABA_TEMPLATES, NAMES, PLACES, OBJECTS). The 15 BABA templates match Figure 14 of
arXiv:2211.00593v1 word for word. ABBA templates swap the first [B] and [A], as the paper and
the code do. Appendix E says places and objects come from "20 common words"; the official code
lists 8 of each, and those are used here.

A prompt stops before the answer: "... [B] gave a [OBJECT] to" (END = the final " to"), so the
model's next token would be the indirect object [A]. [B] is the subject S, which occurs twice
(S1, S2); [A] is the indirect object IO.

Positions are decided here, from the tokenizer's own output, and checked:
  - every filler (name, place, object) is one token with its leading space, so all prompts of
    one template have the same length and the same positions;
  - IO occurs exactly once and S exactly twice in the token ids;
  - the last token is " to".
Fillers that are not one token are dropped before sampling (and the drop is reported).

Control (`ioi_abc_v1`, a paired pABC-style control): the same template, seed, IO, S1, place
and object as the IOI prompt it pairs with, but the S2 slot holds a third name C, so no name is
repeated. Token positions are identical (one-token names), so every position detector reads the
same cells and only the duplication differs. The paper's pABC draws three fresh names; keeping
IO and S1 makes the comparison paired, like the repeated-token control.

No torch here: the tokenizer is passed in (the adapter's tokenize and token_strings).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cirqloom_analysis.detectors.masks import check_integer
from cirqloom_analysis.prompts.repeated import Prompt

IOI_SUITE = "ioi_templates_v1"
IOI_ABC_SUITE = "ioi_abc_v1"
_SALT = 20_221_101

SOURCE = (
    "Wang et al. (2022), arXiv:2211.00593v1, Appendix E / Figure 14; templates and word lists "
    "from redwoodresearch/Easy-Transformer easy_transformer/ioi_dataset.py @ e168301"
)

BABA_TEMPLATES = (
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",  # noqa: E501
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
)

NAMES = (
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda",
    "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon",
    "Jason", "Justin", "Sarah", "William", "Jonathan", "Stephanie", "Brian", "Nicole",
    "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa", "Kevin",
    "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren", "Amber",
    "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle",
    "Tiffany", "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob",
    "Stephen", "Patrick", "Sean", "Erin", "Jamie", "Kelly", "Samantha", "Nathan", "Sara",
    "Dustin", "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory", "Erica",
    "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander",
    "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney", "Christine", "Alicia",
    "Cody", "Allison", "Bradley", "Samuel",
)
PLACES = ("store", "garden", "restaurant", "school", "hospital", "office", "house", "station")
OBJECTS = ("ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack")

Pattern = Literal["BABA", "ABBA"]
POSITION_LABELS = ("IO", "S1", "S1+1", "S2", "END")

Tokenize = Callable[[str], np.ndarray]
"""Text → int64 ids with BOS at position 0 (the adapter's tokenize)."""


def abba(template: str) -> str:
    """Swap the first [B] and the first [A] (official code: ABBA_TEMPLATES)."""
    first_b, first_a = template.index("[B]"), template.index("[A]")
    if first_b > first_a:
        raise ValueError(f"not a BABA template: {template!r}")
    return (template[:first_b] + "[A]" + template[first_b + 3:first_a] + "[B]"
            + template[first_a + 3:])


def template(index: int, pattern: Pattern) -> str:
    check_integer("template index", index)
    if not 0 <= index < len(BABA_TEMPLATES):
        raise ValueError(f"template index must be in 0..{len(BABA_TEMPLATES) - 1}, got {index}")
    if pattern not in ("BABA", "ABBA"):
        raise ValueError(f"pattern must be BABA or ABBA, got {pattern!r}")
    base = BABA_TEMPLATES[index]
    return base if pattern == "BABA" else abba(base)


def group_id(index: int, pattern: Pattern) -> str:
    return f"{pattern}-t{index:02d}"


@dataclass(frozen=True)
class Vocabulary:
    """Fillers that are one token (with their leading space) for this tokenizer."""

    names: tuple[str, ...]
    places: tuple[str, ...]
    objects: tuple[str, ...]
    dropped: tuple[str, ...]


def single_token_vocabulary(tokenize: Tokenize) -> Vocabulary:
    """Keep only fillers that the tokenizer maps to exactly one token after a space."""

    def one_token(word: str) -> bool:
        return tokenize(f" {word}").shape[0] == 2  # BOS + one token

    keep = {
        kind: tuple(w for w in words if one_token(w))
        for kind, words in (("names", NAMES), ("places", PLACES), ("objects", OBJECTS))
    }
    dropped = tuple(w for w in (*NAMES, *PLACES, *OBJECTS) if not one_token(w))
    vocab = Vocabulary(keep["names"], keep["places"], keep["objects"], dropped)
    if len(vocab.names) < 2 or not vocab.places or not vocab.objects:
        raise ValueError("too few one-token fillers for this tokenizer")
    return vocab


def fill(text: str, io: str, s: str, place: str, obj: str, s2: str | None = None) -> str:
    """The prompt text, ending just before the final [A] (the answer).

    `s2` replaces the name in the S2 slot (the second [B]) for the ABC control.
    """
    if not text.endswith(" to [A]"):
        raise ValueError(f"template must end with ' to [A]': {text!r}")
    prompt = text[: -len(" [A]")]
    if s2 is not None:
        second = prompt.index("[B]", prompt.index("[B]") + 3)
        prompt = prompt[:second] + "[C]" + prompt[second + 3:]
    return (prompt.replace("[A]", io).replace("[B]", s).replace("[C]", s2 or "")
            .replace("[PLACE]", place).replace("[OBJECT]", obj))


def locate(token_ids: np.ndarray, io_id: int, s_id: int, end_token: str,
           token_strings: Callable[[np.ndarray], list[str]]) -> dict[str, int]:
    """Named positions in one tokenized prompt, checked against the IOI structure."""
    io_at = np.flatnonzero(token_ids == io_id)
    s_at = np.flatnonzero(token_ids == s_id)
    problems = []
    if io_at.size != 1:
        problems.append(f"IO occurs {io_at.size} times, expected 1")
    if s_at.size != 2:
        problems.append(f"S occurs {s_at.size} times, expected 2")
    last = token_strings(token_ids[-1:])[0]
    if last != end_token:
        problems.append(f"last token is {last!r}, expected {end_token!r}")
    if problems:
        raise ValueError("; ".join(problems))
    s1, s2 = int(s_at[0]), int(s_at[1])
    return {"IO": int(io_at[0]), "S1": s1, "S1+1": s1 + 1, "S2": s2,
            "END": int(token_ids.shape[0] - 1)}


def locate_abc(token_ids: np.ndarray, io_id: int, s_id: int, c_id: int, end_token: str,
               token_strings: Callable[[np.ndarray], list[str]]) -> dict[str, int]:
    """Positions in an ABC control prompt: IO, S (S1) and C (in the S2 slot) once each."""
    found = {label: np.flatnonzero(token_ids == i) for label, i in
             (("IO", io_id), ("S1", s_id), ("S2", c_id))}
    problems = [f"{label} occurs {at.size} times, expected 1"
                for label, at in found.items() if at.size != 1]
    last = token_strings(token_ids[-1:])[0]
    if last != end_token:
        problems.append(f"last token is {last!r}, expected {end_token!r}")
    if problems:
        raise ValueError("; ".join(problems))
    s1 = int(found["S1"][0])
    return {"IO": int(found["IO"][0]), "S1": s1, "S1+1": s1 + 1, "S2": int(found["S2"][0]),
            "END": int(token_ids.shape[0] - 1)}


def ioi_prompt(
    index: int,
    pattern: Pattern,
    seed: int,
    tokenize: Tokenize,
    token_strings: Callable[[np.ndarray], list[str]],
    vocab: Vocabulary,
    control: bool = False,
) -> Prompt:
    """One IOI prompt, or its paired ABC control. Draws are seeded per (template, pattern, seed),
    and the control's third name is drawn after the shared ones, so both share IO, S1, place and
    object."""
    check_integer("seed", seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    rng = np.random.default_rng([_SALT, index, 0 if pattern == "BABA" else 1, seed])
    io, s = (str(n) for n in rng.choice(vocab.names, size=2, replace=False))
    place = str(rng.choice(vocab.places))
    obj = str(rng.choice(vocab.objects))
    c = str(rng.choice([n for n in vocab.names if n not in (io, s)])) if control else None
    text = fill(template(index, pattern), io=io, s=s, place=place, obj=obj, s2=c)

    ids = tokenize(text).astype(np.int64)
    io_id = int(tokenize(f" {io}")[1])
    s_id = int(tokenize(f" {s}")[1])
    try:
        if c is None:
            positions = locate(ids, io_id, s_id, " to", token_strings)
        else:
            positions = locate_abc(ids, io_id, s_id, int(tokenize(f" {c}")[1]), " to",
                                   token_strings)
    except ValueError as error:
        raise ValueError(f"IOI prompt {text!r}: {error}") from None
    group = group_id(index, pattern)
    suite = IOI_ABC_SUITE if control else IOI_SUITE
    return Prompt(
        id=f"{suite}:{group}:seed{seed}",
        suite=suite,
        repeat_len=0,
        seed=seed,
        token_ids=ids,
        text=text,
        positions=positions,
        group=group,
    )


def build_ioi_suite(
    index: int,
    pattern: Pattern,
    seeds: list[int],
    tokenize: Tokenize,
    token_strings: Callable[[np.ndarray], list[str]],
    control: bool = False,
) -> list[Prompt]:
    """One experiment's prompts: one template and pattern, one prompt per seed (or the paired
    ABC control prompts with `control=True`).

    Raises unless every prompt has the same length and the same named positions, because the
    detectors of this suite read one set of positions per experiment.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    vocab = single_token_vocabulary(tokenize)
    prompts = [ioi_prompt(index, pattern, seed, tokenize, token_strings, vocab, control)
               for seed in seeds]
    first = prompts[0]
    for p in prompts[1:]:
        if p.token_ids.shape != first.token_ids.shape or p.positions != first.positions:
            raise ValueError(
                f"{p.id}: length/positions {p.token_ids.shape[0]}/{p.positions} differ from "
                f"{first.id}: {first.token_ids.shape[0]}/{first.positions}"
            )
    return prompts
