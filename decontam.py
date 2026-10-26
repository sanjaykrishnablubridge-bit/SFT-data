"""Benchmark decontamination: drop SFT examples that contain most of an eval item.

An example is contaminated if it shares at least `min_overlap` of some eval
item's word 13-grams (the Tulu 3 approach). Matching on a single shared
13-gram is too aggressive: IFEval prompts are built from fixed instruction
templates ("add a postscript starting with P.S."), and constraint-following
SFT data legitimately reuses that wording.

Eval texts come from the public test/validation sets of the benchmarks we
report. Items shorter than n words can't be matched this way and are counted
in the report as uncovered.
"""
import pickle
import re
from collections import Counter, defaultdict

from datasets import load_dataset

from data import CACHE, file_lock, fingerprint

# (hf_id, config, split, function(row) -> text). Kept in sync with eval tasks.
EVAL_SETS = {
    "gsm8k": ("openai/gsm8k", "main", "test", lambda r: r["question"]),
    "ifeval": ("google/IFEval", None, "train", lambda r: r["prompt"]),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", lambda r: r["question"]),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", lambda r: r["question"]),
    "hellaswag": ("Rowan/hellaswag", None, "validation", lambda r: r["ctx"]),
    "openbookqa": ("allenai/openbookqa", "main", "test", lambda r: r["question_stem"]),
    "piqa": ("baber/piqa", None, "validation", lambda r: r["goal"]),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", lambda r: r["sentence"]),
    "mmlu": ("cais/mmlu", "all", "test", lambda r: r["question"]),
}

_WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return _WORD.findall(text.lower())


def ngrams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


class Decontaminator:
    def __init__(self, tasks, n=13, min_overlap=0.5):
        self.n, self.min_overlap = n, min_overlap
        path = CACHE / f"evalgrams_{fingerprint([sorted(tasks), n, 2])}.pkl"
        with file_lock(path):  # concurrent runs build the index once
            self._load_or_build(tasks, n, path)

    def _load_or_build(self, tasks, n, path):
        if path.exists():
            self.index, self.sizes, self.coverage = pickle.loads(path.read_bytes())
            return
        index, self.sizes, self.coverage = defaultdict(list), [], {}
        for task in tasks:
            if task not in EVAL_SETS:
                self.coverage[task] = "no eval set registered in decontam.py"
                continue
            hf_id, conf, split, get = EVAL_SETS[task]
            try:
                ds = load_dataset(hf_id, conf, split=split)
            except Exception as e:  # keep going; the report shows what wasn't checked
                self.coverage[task] = f"could not load ({type(e).__name__})"
                continue
            short = 0
            for row in ds:
                grams = ngrams(words(get(row)), n)
                if not grams:
                    short += 1
                    continue
                for g in grams:
                    index[g].append(len(self.sizes))
                self.sizes.append(len(grams))
            self.coverage[task] = f"{len(ds) - short}/{len(ds)} items checkable"
        self.index = dict(index)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps((self.index, self.sizes, self.coverage)))

    def hit(self, messages):
        grams = ngrams(words(" ".join(m["content"] for m in messages)), self.n)
        counts = Counter(item for g in grams if g in self.index for item in self.index[g])
        return any(c >= self.min_overlap * self.sizes[item] for item, c in counts.items())
