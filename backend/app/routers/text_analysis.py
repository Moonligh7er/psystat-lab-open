"""Text / NLP analysis for behavioral science research.

Endpoints: sentiment, topic-model, liwc-style, word-frequency, readability, content-coding.
Uses only stdlib + numpy + scipy + sklearn (already in project).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats
from sklearn.decomposition import NMF as SklearnNMF

router = APIRouter()

# ═══════════════════════════════════════════════════════════════════════
#  LEXICONS & WORD LISTS
# ═══════════════════════════════════════════════════════════════════════

STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "like", "ll", "me", "might",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan't",
    "she", "should", "shouldn't", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "ve", "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "also", "been", "being", "come", "else", "ever", "every", "go", "going",
    "gone", "got", "gotten", "however", "instead", "just", "keep", "kept",
    "know", "known", "later", "less", "lot", "lots", "made", "make",
    "making", "many", "may", "maybe", "much", "must", "need", "new",
    "nothing", "often", "old", "one", "ones", "part", "per", "put",
    "quite", "really", "right", "said", "say", "see", "seem", "seemed",
    "seems", "shall", "since", "something", "still", "such", "sure",
    "take", "tell", "thing", "things", "think", "though", "thus", "today",
    "together", "told", "took", "try", "two", "upon", "us", "use", "used",
    "using", "want", "way", "well", "went", "what", "whether", "without",
    "work", "works", "world", "would", "yet", "already", "always", "among",
    "another", "around", "away", "back", "become", "best", "better", "big",
    "came", "called", "d", "m", "even", "found", "gave", "give", "great",
    "hand", "high", "home", "house", "last", "left", "life", "little",
    "long", "look", "man", "men", "might", "never", "next", "nothing",
    "number", "own", "people", "place", "point", "rather", "set", "small",
    "state", "still", "three", "time", "turn", "turned", "year", "years",
}

# ── Sentiment lexicon (~200 words) ────────────────────────────────────
# Positive words → valence (+0.5 to +1.0)
_POS: dict[str, float] = {
    "good": 0.7, "great": 0.8, "excellent": 0.9, "wonderful": 0.9,
    "fantastic": 0.9, "amazing": 0.9, "love": 0.8, "loved": 0.8,
    "lovely": 0.7, "best": 0.8, "happy": 0.8, "happily": 0.7,
    "happiness": 0.8, "joy": 0.8, "joyful": 0.8, "beautiful": 0.8,
    "brilliant": 0.8, "awesome": 0.8, "outstanding": 0.9,
    "superb": 0.9, "perfect": 0.9, "nice": 0.6, "pleasant": 0.6,
    "enjoy": 0.7, "enjoyed": 0.7, "enjoyable": 0.7, "like": 0.5,
    "liked": 0.5, "pleased": 0.7, "pleasure": 0.7, "glad": 0.7,
    "delightful": 0.8, "delight": 0.8, "cheerful": 0.7, "positive": 0.6,
    "exciting": 0.7, "excited": 0.7, "enthusiasm": 0.7, "enthusiastic": 0.7,
    "satisfied": 0.7, "satisfying": 0.7, "impressive": 0.7,
    "remarkable": 0.8, "incredible": 0.8, "marvelous": 0.8,
    "splendid": 0.8, "terrific": 0.8, "fabulous": 0.8, "magnificent": 0.9,
    "glorious": 0.8, "triumph": 0.8, "success": 0.7, "successful": 0.7,
    "win": 0.7, "winning": 0.7, "won": 0.7, "hope": 0.6, "hopeful": 0.6,
    "trust": 0.6, "kind": 0.6, "kindness": 0.7, "gentle": 0.6,
    "generous": 0.7, "warm": 0.6, "warmth": 0.6, "comfort": 0.6,
    "comfortable": 0.6, "safe": 0.6, "peaceful": 0.7, "calm": 0.6,
    "brave": 0.7, "courage": 0.7, "proud": 0.7, "pride": 0.7,
    "inspire": 0.7, "inspired": 0.7, "inspiring": 0.7, "grateful": 0.7,
    "thankful": 0.7, "appreciate": 0.7, "wonderful": 0.9,
    "sweet": 0.6, "charming": 0.7, "elegant": 0.7, "fun": 0.7,
    "funny": 0.6, "laugh": 0.6, "laughing": 0.6, "smile": 0.6,
    "smiling": 0.6, "bright": 0.6, "vibrant": 0.7, "thrive": 0.7,
    "flourish": 0.8, "prosper": 0.7, "reward": 0.7, "rewarding": 0.7,
    "benefit": 0.6, "beneficial": 0.6, "improve": 0.6, "improved": 0.6,
    "heal": 0.6, "healthy": 0.6, "strong": 0.6, "strength": 0.6,
    "support": 0.6, "supportive": 0.6, "care": 0.6, "caring": 0.6,
}

# Negative words → valence (-0.5 to -1.0)
_NEG: dict[str, float] = {
    "bad": -0.7, "terrible": -0.9, "horrible": -0.9, "awful": -0.9,
    "worst": -0.9, "hate": -0.8, "hated": -0.8, "hatred": -0.9,
    "ugly": -0.7, "disgusting": -0.9, "disgust": -0.8, "nasty": -0.8,
    "dreadful": -0.8, "pathetic": -0.8, "miserable": -0.8,
    "sad": -0.7, "sadly": -0.7, "sadness": -0.7, "unhappy": -0.7,
    "depressed": -0.8, "depressing": -0.8, "depression": -0.8,
    "angry": -0.8, "anger": -0.8, "furious": -0.9, "rage": -0.9,
    "fear": -0.8, "afraid": -0.8, "scared": -0.8, "terrified": -0.9,
    "anxious": -0.7, "anxiety": -0.7, "worry": -0.6, "worried": -0.6,
    "stress": -0.7, "stressed": -0.7, "stressful": -0.7,
    "pain": -0.7, "painful": -0.7, "hurt": -0.7, "hurting": -0.7,
    "suffer": -0.8, "suffering": -0.8, "agony": -0.9,
    "fail": -0.7, "failed": -0.7, "failure": -0.8, "lose": -0.7,
    "lost": -0.6, "loss": -0.7, "losing": -0.7,
    "boring": -0.6, "bored": -0.6, "dull": -0.6, "mediocre": -0.5,
    "disappoint": -0.7, "disappointed": -0.7, "disappointing": -0.7,
    "frustrate": -0.7, "frustrated": -0.7, "frustrating": -0.7,
    "annoy": -0.6, "annoyed": -0.6, "annoying": -0.6, "irritate": -0.6,
    "irritated": -0.6, "irritating": -0.6,
    "weak": -0.6, "weakness": -0.6, "helpless": -0.7, "hopeless": -0.8,
    "lonely": -0.7, "loneliness": -0.7, "alone": -0.5,
    "guilt": -0.7, "guilty": -0.7, "shame": -0.7, "ashamed": -0.7,
    "embarrass": -0.7, "embarrassed": -0.7, "embarrassing": -0.7,
    "cruel": -0.8, "cruelty": -0.8, "harm": -0.7, "harmful": -0.7,
    "damage": -0.7, "destroy": -0.8, "destruction": -0.8,
    "death": -0.8, "dead": -0.7, "die": -0.7, "dying": -0.7,
    "sick": -0.6, "illness": -0.6, "disease": -0.6,
    "wrong": -0.6, "error": -0.5, "mistake": -0.5,
    "reject": -0.7, "rejected": -0.7, "rejection": -0.7,
    "neglect": -0.7, "abuse": -0.9, "violent": -0.9, "violence": -0.9,
    "threat": -0.7, "threaten": -0.7, "danger": -0.7, "dangerous": -0.7,
    "risk": -0.5, "problem": -0.5, "trouble": -0.6, "crisis": -0.7,
    "conflict": -0.6, "struggle": -0.6, "difficult": -0.5,
    "difficulty": -0.5, "hard": -0.4, "negative": -0.6,
    "toxic": -0.8, "poison": -0.8, "corrupt": -0.8, "evil": -0.9,
    "wicked": -0.8, "hostile": -0.8, "hostile": -0.8,
}

SENTIMENT_LEXICON: dict[str, float] = {**_POS, **_NEG}

NEGATORS: set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "barely", "hardly",
    "scarcely", "seldom", "rarely",
}

INTENSIFIERS: dict[str, float] = {
    "very": 1.3, "extremely": 1.5, "incredibly": 1.5, "absolutely": 1.4,
    "really": 1.25, "truly": 1.25, "so": 1.2, "utterly": 1.5,
    "highly": 1.3, "deeply": 1.3, "remarkably": 1.3, "exceptionally": 1.4,
    "especially": 1.25, "particularly": 1.25, "terribly": 1.3,
    "enormously": 1.4, "totally": 1.3, "completely": 1.35,
    "quite": 1.15, "rather": 1.1, "fairly": 1.1, "somewhat": 0.85,
    "slightly": 0.7, "barely": 0.6, "hardly": 0.6, "little": 0.8,
}

# ── LIWC-style categories (~30-50 words each) ────────────────────────
LIWC_CATEGORIES: dict[str, list[str]] = {
    "positive_emotion": [
        "love", "nice", "sweet", "happy", "joy", "hope", "proud", "glad",
        "cheerful", "delight", "enjoy", "wonderful", "great", "excellent",
        "good", "kind", "warm", "gentle", "caring", "grateful", "thankful",
        "pleased", "satisfied", "content", "excited", "enthusiasm", "thrill",
        "bliss", "comfort", "amuse", "laugh", "smile", "fun", "paradise",
        "bless", "heaven", "inspire", "admire", "adore", "treasure",
        "celebrate", "triumph", "elat", "cheer", "optimis",
    ],
    "negative_emotion": [
        "hate", "ugly", "nasty", "awful", "terrible", "horrible", "hurt",
        "suffer", "pain", "agony", "miserable", "wretched", "dread",
        "despair", "hopeless", "helpless", "torment", "anguish", "grief",
        "sorrow", "mourn", "gloom", "doom", "tragic", "devastat",
        "heartbreak", "bitter", "resent", "contempt", "disgust", "loathe",
        "abhor", "detest", "spite", "hostil", "cruel", "harsh", "brutal",
    ],
    "anger": [
        "angry", "anger", "mad", "furious", "rage", "hostile", "resent",
        "irritat", "annoy", "frustrat", "outrag", "infuriat", "wrath",
        "aggress", "violent", "fight", "attack", "destroy", "revenge",
        "spite", "contempt", "scorn", "hate", "bitter", "jealous",
        "envious", "betray", "offend", "insult", "yell", "scream",
        "explod", "tantrum", "bully", "threat",
    ],
    "anxiety": [
        "anxious", "anxiety", "worry", "worried", "nervous", "tense",
        "stress", "scared", "afraid", "fear", "dread", "panic", "alarm",
        "apprehens", "uneasy", "restless", "fright", "terrif", "phobia",
        "obsess", "compuls", "paranoi", "insecur", "doubt", "uncertain",
        "overwhelm", "pressur", "strain", "threat", "risk", "danger",
        "vulnerab", "exposed", "helpless", "trembl",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depress", "down", "blue", "cry",
        "crying", "tears", "weep", "mourn", "grief", "sorrow", "loss",
        "miss", "lonely", "loneliness", "alone", "abandon", "reject",
        "heartbreak", "disappoint", "hopeless", "despair", "gloom",
        "melanchol", "somber", "empty", "numb", "pain", "ache",
        "regret", "remorse", "resign", "defeat", "suffer",
    ],
    "social": [
        "friend", "family", "people", "person", "group", "team",
        "community", "social", "relationship", "together", "share",
        "talk", "speak", "listen", "communicate", "connect", "bond",
        "belong", "member", "partner", "colleague", "neighbor",
        "companion", "ally", "peer", "crowd", "society", "network",
        "gather", "meet", "visit", "invite", "greet", "help", "cooperat",
        "collabor", "trust", "respect", "compan",
    ],
    "cognitive": [
        "think", "thought", "know", "understand", "realize", "believe",
        "consider", "reason", "logic", "analyz", "evaluat", "assess",
        "interpret", "conclud", "infer", "deduc", "hypothes", "theory",
        "concept", "idea", "learn", "study", "research", "discover",
        "wonder", "question", "curious", "reflect", "ponder", "contempl",
        "deliberat", "decide", "judg", "perceiv", "recogniz", "aware",
        "conscious", "mindful", "comprehend", "grasp",
    ],
    "causal": [
        "because", "cause", "effect", "therefore", "thus", "hence",
        "consequently", "result", "lead", "produce", "create", "generate",
        "trigger", "induce", "influence", "impact", "determine", "depend",
        "reason", "explain", "why", "how", "origin", "source", "basis",
        "factor", "contribut", "attribut", "due", "since", "so",
    ],
    "insight": [
        "understand", "realize", "discover", "insight", "aware", "clarity",
        "reveal", "recognize", "perceive", "grasp", "comprehend", "learn",
        "epiphany", "enlighten", "illuminate", "uncover", "detect",
        "notice", "observe", "discern", "fathom", "appreciate", "dawn",
        "breakthrough", "revelation", "eureka", "aha", "knew", "found",
        "sense", "intuit",
    ],
    "tentative": [
        "maybe", "perhaps", "possibly", "might", "could", "seem",
        "appear", "suggest", "guess", "suppose", "wonder", "hope",
        "likely", "unlikely", "probable", "possible", "uncertain",
        "unclear", "ambiguous", "doubt", "hesitat", "tentativ", "if",
        "whether", "approximately", "roughly", "around", "almost",
        "nearly", "somewhat", "fairly", "partly", "sort of", "kind of",
    ],
    "certainty": [
        "certain", "sure", "definite", "absolute", "clear", "obvious",
        "always", "never", "every", "all", "none", "must", "undoubtedly",
        "certainly", "definitely", "precisely", "exactly", "truly",
        "indeed", "fact", "proven", "guarantee", "inevitable", "without doubt",
        "unquestion", "indisputab", "conclusive", "confirm", "assure",
        "positive", "commit",
    ],
    "achievement": [
        "achieve", "accomplish", "success", "win", "goal", "earn",
        "attain", "complete", "finish", "master", "excel", "improve",
        "progress", "advance", "develop", "grow", "thrive", "prosper",
        "overcome", "conquer", "triumph", "reward", "prize", "honor",
        "award", "rank", "compet", "perform", "productiv", "efficient",
        "effective", "skill", "talent", "capab", "proficien",
    ],
    "power": [
        "power", "control", "authority", "command", "lead", "dominate",
        "influence", "force", "strong", "strength", "might", "superior",
        "boss", "chief", "govern", "rule", "direct", "manage", "order",
        "demand", "require", "insist", "assert", "bold", "confident",
        "decisive", "firm", "determin", "ambitious", "drive", "compel",
        "enforce", "pressure", "intimidat", "overpower",
    ],
    "affiliation": [
        "friend", "family", "love", "together", "share", "team", "group",
        "belong", "connect", "bond", "close", "intimate", "companion",
        "partner", "ally", "loyal", "devot", "commit", "attach", "trust",
        "depend", "support", "help", "cooperat", "collabor", "unite",
        "solidarity", "communal", "mutual", "reciproc", "empathy",
        "compassion", "sympathy", "care", "nurture", "comfort",
    ],
    "self_reference": [
        "i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd",
        "im",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENTENCE_RE = re.compile(r"[.!?]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, keep contractions."""
    return _TOKEN_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter based on vowel clusters."""
    word = word.lower().rstrip("e")
    if not word:
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


def _safe_float(v) -> float:
    """Convert numpy / nan to JSON-safe float."""
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return round(f, 4)


def _remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def _stem_match(word: str, patterns: list[str]) -> bool:
    """Check if word starts with any pattern (prefix/stem matching)."""
    for p in patterns:
        if word == p or word.startswith(p):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class SentimentRequest(BaseModel):
    texts: list[str]


class TopicModelRequest(BaseModel):
    texts: list[str]
    n_topics: int = 5
    method: str = "nmf"  # "nmf" or "lsa"


class LIWCRequest(BaseModel):
    texts: list[str]
    groups: Optional[list[int]] = None  # optional group variable for comparison


class WordFrequencyRequest(BaseModel):
    texts: list[str]
    top_n: int = 50
    target_words: Optional[list[str]] = None  # for KWIC
    kwic_window: int = 5


class ReadabilityRequest(BaseModel):
    texts: list[str]


class ContentCodingRequest(BaseModel):
    texts: list[str]
    coding_scheme: dict[str, list[str]]  # category_name → list of keyword/regex patterns


# ═══════════════════════════════════════════════════════════════════════
#  1. SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

@router.post("/sentiment")
def sentiment_analysis(req: SentimentRequest):
    """VADER-like sentiment analysis with negation and intensifier handling."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")

    results: list[dict] = []
    all_sentiment_words: list[str] = []

    for text in req.texts:
        tokens = _tokenize(text)
        word_count = len(tokens)
        if word_count == 0:
            results.append({
                "positive": 0.0, "negative": 0.0, "neutral": 1.0,
                "compound": 0.0, "word_count": 0, "sentiment_words": [],
            })
            continue

        raw_scores: list[float] = []
        sent_words: list[str] = []

        for i, tok in enumerate(tokens):
            valence = SENTIMENT_LEXICON.get(tok)
            if valence is None:
                continue

            # check intensifier in preceding word
            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                valence *= INTENSIFIERS[tokens[i - 1]]

            # check negation within 3-word window before
            negated = False
            for j in range(max(0, i - 3), i):
                if tokens[j] in NEGATORS:
                    negated = True
                    break
            if negated:
                valence *= -0.75  # flip with slight dampening

            raw_scores.append(valence)
            sent_words.append(tok)

        all_sentiment_words.extend(sent_words)

        if raw_scores:
            pos_sum = sum(s for s in raw_scores if s > 0)
            neg_sum = sum(s for s in raw_scores if s < 0)
            total = pos_sum + abs(neg_sum)
            pos_score = _safe_float(pos_sum / total) if total else 0.0
            neg_score = _safe_float(abs(neg_sum) / total) if total else 0.0
            neutral_score = _safe_float(1.0 - (len(raw_scores) / word_count))
            # compound: normalise sum to [-1, 1] using tanh-like transform
            raw_sum = sum(raw_scores)
            compound = _safe_float(raw_sum / math.sqrt(raw_sum ** 2 + 15))
        else:
            pos_score = neg_score = 0.0
            neutral_score = 1.0
            compound = 0.0

        results.append({
            "positive": pos_score,
            "negative": neg_score,
            "neutral": _safe_float(max(neutral_score, 0.0)),
            "compound": compound,
            "word_count": word_count,
            "sentiment_words": sent_words,
        })

    compounds = [r["compound"] for r in results]
    compounds_arr = np.array(compounds)

    # distribution
    n_pos = sum(1 for c in compounds if c > 0.05)
    n_neg = sum(1 for c in compounds if c < -0.05)
    n_neu = len(compounds) - n_pos - n_neg

    word_counter = Counter(all_sentiment_words)

    return {
        "per_text": results,
        "aggregate": {
            "mean_compound": _safe_float(compounds_arr.mean()),
            "sd_compound": _safe_float(compounds_arr.std(ddof=1)) if len(compounds) > 1 else 0.0,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_neutral": n_neu,
            "distribution": {
                "positive_pct": _safe_float(n_pos / len(compounds)),
                "negative_pct": _safe_float(n_neg / len(compounds)),
                "neutral_pct": _safe_float(n_neu / len(compounds)),
            },
            "most_common_sentiment_words": word_counter.most_common(20),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  2. TOPIC MODELING (NMF / LSA)
# ═══════════════════════════════════════════════════════════════════════

def _build_tfidf(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build TF-IDF matrix from texts. Returns (matrix, vocab)."""
    # tokenize & remove stopwords
    docs = [_remove_stopwords(_tokenize(t)) for t in texts]

    # build vocabulary
    vocab_counter: Counter = Counter()
    for doc in docs:
        vocab_counter.update(set(doc))  # document frequency

    # filter: appear in at least 2 docs (if enough docs), not too frequent
    n_docs = len(docs)
    min_df = 2 if n_docs >= 4 else 1
    max_df_ratio = 0.85
    vocab = sorted([
        w for w, c in vocab_counter.items()
        if c >= min_df and c / n_docs <= max_df_ratio
    ])

    if not vocab:
        # fallback: use all words appearing at least once
        vocab = sorted(vocab_counter.keys())

    if not vocab:
        return np.zeros((n_docs, 1)), ["<empty>"]

    word2idx = {w: i for i, w in enumerate(vocab)}

    # term frequency matrix
    tf = np.zeros((n_docs, len(vocab)))
    for di, doc in enumerate(docs):
        for tok in doc:
            if tok in word2idx:
                tf[di, word2idx[tok]] += 1
        # normalise TF by doc length
        doc_len = len(doc)
        if doc_len > 0:
            tf[di] /= doc_len

    # IDF
    df = np.array([vocab_counter[w] for w in vocab], dtype=float)
    idf = np.log((n_docs + 1) / (df + 1)) + 1  # smoothed IDF

    tfidf = tf * idf
    return tfidf, vocab


def _topic_coherence_pmi(topic_words: list[str], docs: list[list[str]], top_n: int = 10) -> float:
    """Compute PMI-based topic coherence for a list of topic words."""
    n_docs = len(docs)
    if n_docs == 0:
        return 0.0

    word_doc_sets: dict[str, set[int]] = defaultdict(set)
    for di, doc in enumerate(docs):
        for w in set(doc):
            word_doc_sets[w].add(di)

    pmi_sum = 0.0
    count = 0
    words = topic_words[:top_n]
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            co = len(word_doc_sets[w1] & word_doc_sets[w2])
            p_w1 = len(word_doc_sets[w1]) / n_docs if w1 in word_doc_sets else 1e-10
            p_w2 = len(word_doc_sets[w2]) / n_docs if w2 in word_doc_sets else 1e-10
            p_co = (co + 1e-10) / n_docs
            pmi_sum += math.log(p_co / (p_w1 * p_w2 + 1e-10))
            count += 1

    return _safe_float(pmi_sum / count) if count > 0 else 0.0


@router.post("/topic-model")
def topic_model(req: TopicModelRequest):
    """Topic modeling via NMF or LSA on TF-IDF matrix."""
    if not req.texts or len(req.texts) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 texts for topic modeling")
    if req.n_topics < 1:
        raise HTTPException(status_code=400, detail="n_topics must be >= 1")

    tfidf, vocab = _build_tfidf(req.texts)
    n_docs, n_terms = tfidf.shape
    n_topics = min(req.n_topics, min(n_docs, n_terms))

    if req.method.lower() == "lsa":
        # Truncated SVD
        try:
            U, S, Vt = np.linalg.svd(tfidf, full_matrices=False)
            topic_term = Vt[:n_topics]  # (n_topics, n_terms)
            doc_topic = U[:, :n_topics] * S[:n_topics]  # (n_docs, n_topics)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SVD failed: {e}")
    else:
        # NMF via sklearn
        try:
            model = SklearnNMF(n_components=n_topics, max_iter=300, random_state=42)
            tfidf_nn = np.maximum(tfidf, 0)  # NMF needs non-negative
            doc_topic = model.fit_transform(tfidf_nn)  # (n_docs, n_topics)
            topic_term = model.components_  # (n_topics, n_terms)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NMF failed: {e}")

    # Extract top words per topic
    docs_tokenized = [_remove_stopwords(_tokenize(t)) for t in req.texts]
    topics = []
    for ti in range(n_topics):
        weights = topic_term[ti]
        top_indices = np.argsort(np.abs(weights))[::-1][:10]
        top_words = [(vocab[idx], _safe_float(weights[idx])) for idx in top_indices]
        label = " + ".join([w for w, _ in top_words[:3]])
        coherence = _topic_coherence_pmi(
            [w for w, _ in top_words], docs_tokenized
        )
        topics.append({
            "topic_id": ti,
            "label": label,
            "top_words": [{"word": w, "weight": wt} for w, wt in top_words],
            "coherence": coherence,
        })

    # Normalise doc-topic distributions
    doc_topic_abs = np.abs(doc_topic)
    row_sums = doc_topic_abs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topic_norm = doc_topic_abs / row_sums

    return {
        "method": req.method.lower(),
        "n_topics": n_topics,
        "n_documents": n_docs,
        "n_terms": n_terms,
        "topics": topics,
        "document_topic_distributions": [
            [_safe_float(v) for v in row] for row in doc_topic_norm
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  3. LIWC-STYLE PSYCHOLOGICAL LANGUAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

@router.post("/liwc-style")
def liwc_analysis(req: LIWCRequest):
    """Psychological language analysis using LIWC-inspired word categories."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")

    category_names = list(LIWC_CATEGORIES.keys())
    per_text: list[dict] = []

    for text in req.texts:
        tokens = _tokenize(text)
        word_count = len(tokens)
        sents = _sentences(text)
        words_per_sentence = word_count / max(len(sents), 1)

        cat_counts: dict[str, int] = {}
        for cat, patterns in LIWC_CATEGORIES.items():
            count = sum(1 for tok in tokens if _stem_match(tok, patterns))
            cat_counts[cat] = count

        cat_pcts = {
            cat: _safe_float(100 * count / word_count) if word_count else 0.0
            for cat, count in cat_counts.items()
        }

        per_text.append({
            "word_count": word_count,
            "words_per_sentence": _safe_float(words_per_sentence),
            "categories": cat_pcts,
        })

    # Aggregate
    n = len(per_text)
    agg_means: dict[str, float] = {}
    agg_sds: dict[str, float] = {}
    cat_arrays: dict[str, np.ndarray] = {}
    for cat in category_names:
        vals = np.array([pt["categories"][cat] for pt in per_text])
        cat_arrays[cat] = vals
        agg_means[cat] = _safe_float(vals.mean())
        agg_sds[cat] = _safe_float(vals.std(ddof=1)) if n > 1 else 0.0

    # Correlations between categories
    correlations: dict[str, dict[str, float]] = {}
    if n > 2:
        for i, c1 in enumerate(category_names):
            correlations[c1] = {}
            for j, c2 in enumerate(category_names):
                if i == j:
                    correlations[c1][c2] = 1.0
                elif j > i:
                    r, _ = sp_stats.pearsonr(cat_arrays[c1], cat_arrays[c2])
                    correlations[c1][c2] = _safe_float(r)
                else:
                    correlations[c1][c2] = correlations[c2][c1]

    result = {
        "per_text": per_text,
        "aggregate": {
            "means": agg_means,
            "sds": agg_sds,
            "correlations": correlations if n > 2 else None,
        },
    }

    # Group comparison (if groups provided)
    if req.groups is not None:
        if len(req.groups) != len(req.texts):
            raise HTTPException(status_code=400, detail="groups length must match texts length")
        unique_groups = sorted(set(req.groups))
        if len(unique_groups) == 2:
            g1, g2 = unique_groups
            comparison: dict[str, dict] = {}
            for cat in category_names:
                vals1 = np.array([
                    per_text[i]["categories"][cat]
                    for i in range(n) if req.groups[i] == g1
                ])
                vals2 = np.array([
                    per_text[i]["categories"][cat]
                    for i in range(n) if req.groups[i] == g2
                ])
                if len(vals1) > 1 and len(vals2) > 1:
                    t_stat, p_val = sp_stats.ttest_ind(vals1, vals2)
                    cohens_d = _safe_float(
                        (vals1.mean() - vals2.mean()) /
                        math.sqrt((vals1.std(ddof=1) ** 2 + vals2.std(ddof=1) ** 2) / 2)
                    ) if (vals1.std(ddof=1) + vals2.std(ddof=1)) > 0 else 0.0
                    comparison[cat] = {
                        f"group_{g1}_mean": _safe_float(vals1.mean()),
                        f"group_{g2}_mean": _safe_float(vals2.mean()),
                        "t_statistic": _safe_float(t_stat),
                        "p_value": _safe_float(p_val),
                        "cohens_d": cohens_d,
                    }
            result["group_comparison"] = comparison

    return result


# ═══════════════════════════════════════════════════════════════════════
#  4. WORD FREQUENCY & COLLOCATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

@router.post("/word-frequency")
def word_frequency(req: WordFrequencyRequest):
    """Word frequency, collocations, type-token ratio, KWIC analysis."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")

    all_tokens: list[str] = []
    clean_tokens: list[str] = []
    for text in req.texts:
        toks = _tokenize(text)
        all_tokens.extend(toks)
        clean_tokens.extend(_remove_stopwords(toks))

    total_words = len(all_tokens)
    if total_words == 0:
        raise HTTPException(status_code=400, detail="No words found in texts")

    # Word frequencies (without stopwords)
    freq = Counter(clean_tokens)
    top_words = [
        {"word": w, "count": c, "pct": _safe_float(100 * c / total_words)}
        for w, c in freq.most_common(req.top_n)
    ]

    # Hapax legomena (words appearing once)
    hapax = [w for w, c in freq.items() if c == 1]

    # Type-token ratio
    n_types = len(set(all_tokens))
    n_tokens = len(all_tokens)
    ttr = _safe_float(n_types / n_tokens) if n_tokens else 0.0

    # Yule's K
    freq_spectrum = Counter(freq.values())  # frequency of frequencies
    m1 = sum(freq.values())
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    yules_k = _safe_float(
        10000 * (m2 - m1) / (m1 * m1)
    ) if m1 > 0 else 0.0

    # Bigrams
    bigrams = Counter()
    for i in range(len(clean_tokens) - 1):
        bigrams[(clean_tokens[i], clean_tokens[i + 1])] += 1
    top_bigrams = [
        {"bigram": f"{b[0]} {b[1]}", "count": c}
        for b, c in bigrams.most_common(req.top_n)
    ]

    # Trigrams
    trigrams = Counter()
    for i in range(len(clean_tokens) - 2):
        trigrams[(clean_tokens[i], clean_tokens[i + 1], clean_tokens[i + 2])] += 1
    top_trigrams = [
        {"trigram": f"{t[0]} {t[1]} {t[2]}", "count": c}
        for t, c in trigrams.most_common(req.top_n)
    ]

    # KWIC (Keyword In Context)
    kwic_results: dict[str, list[dict]] = {}
    if req.target_words:
        for target in req.target_words:
            target_lower = target.lower()
            contexts = []
            for text in req.texts:
                toks = _tokenize(text)
                for i, tok in enumerate(toks):
                    if tok == target_lower:
                        left = " ".join(toks[max(0, i - req.kwic_window):i])
                        right = " ".join(toks[i + 1:i + 1 + req.kwic_window])
                        contexts.append({
                            "left": left,
                            "keyword": tok,
                            "right": right,
                        })
            kwic_results[target] = contexts

    # Word cloud data (top 100)
    word_cloud = [
        {"word": w, "frequency": c}
        for w, c in freq.most_common(100)
    ]

    return {
        "total_words": total_words,
        "unique_words": n_types,
        "top_words": top_words,
        "hapax_legomena_count": len(hapax),
        "type_token_ratio": ttr,
        "yules_k": yules_k,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "kwic": kwic_results if req.target_words else None,
        "word_cloud_data": word_cloud,
    }


# ═══════════════════════════════════════════════════════════════════════
#  5. READABILITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def _readability_scores(text: str) -> dict:
    """Compute all readability indices for a single text."""
    tokens = _tokenize(text)
    sents = _sentences(text)
    word_count = len(tokens)
    sent_count = max(len(sents), 1)
    syllable_counts = [_count_syllables(w) for w in tokens]
    total_syllables = sum(syllable_counts)
    polysyllable_count = sum(1 for s in syllable_counts if s >= 3)

    avg_sentence_len = word_count / sent_count
    avg_syllables_per_word = total_syllables / max(word_count, 1)

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word

    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59

    # Gunning Fog Index
    fog = 0.4 * (avg_sentence_len + 100 * (polysyllable_count / max(word_count, 1)))

    # Coleman-Liau Index
    avg_chars_per_100 = sum(len(w) for w in tokens) / max(word_count, 1) * 100
    avg_sents_per_100 = sent_count / max(word_count, 1) * 100
    cli = 0.0588 * avg_chars_per_100 - 0.296 * avg_sents_per_100 - 15.8

    # SMOG Index
    smog = 1.0430 * math.sqrt(polysyllable_count * (30 / max(sent_count, 1))) + 3.1291

    # Automated Readability Index
    avg_chars_per_word = sum(len(w) for w in tokens) / max(word_count, 1)
    ari = 4.71 * avg_chars_per_word + 0.5 * avg_sentence_len - 21.43

    # Interpretation
    if fkgl <= 5:
        interpretation = "Elementary school level"
    elif fkgl <= 8:
        interpretation = "Middle school level"
    elif fkgl <= 12:
        interpretation = "High school level"
    elif fkgl <= 16:
        interpretation = "College level"
    else:
        interpretation = "Graduate / professional level"

    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "syllable_count": total_syllables,
        "polysyllable_count": polysyllable_count,
        "flesch_reading_ease": _safe_float(fre),
        "flesch_kincaid_grade": _safe_float(fkgl),
        "gunning_fog": _safe_float(fog),
        "coleman_liau": _safe_float(cli),
        "smog": _safe_float(smog),
        "ari": _safe_float(ari),
        "interpretation": interpretation,
        "grade_level": _safe_float(fkgl),
    }


@router.post("/readability")
def readability_analysis(req: ReadabilityRequest):
    """Compute readability metrics for each text and aggregate."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")

    per_text = [_readability_scores(t) for t in req.texts]

    # Aggregate
    metrics = [
        "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
        "coleman_liau", "smog", "ari",
    ]
    agg: dict[str, dict[str, float]] = {}
    for m in metrics:
        vals = np.array([pt[m] for pt in per_text])
        agg[m] = {
            "mean": _safe_float(vals.mean()),
            "sd": _safe_float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": _safe_float(vals.min()),
            "max": _safe_float(vals.max()),
        }

    return {
        "per_text": per_text,
        "aggregate": agg,
    }


# ═══════════════════════════════════════════════════════════════════════
#  6. AUTOMATED CONTENT CODING
# ═══════════════════════════════════════════════════════════════════════

@router.post("/content-coding")
def content_coding(req: ContentCodingRequest):
    """Match texts against a user-defined coding scheme; chi-square for independence."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")
    if not req.coding_scheme:
        raise HTTPException(status_code=400, detail="coding_scheme must not be empty")

    categories = list(req.coding_scheme.keys())
    n_texts = len(req.texts)
    n_cats = len(categories)

    # Build compiled patterns
    compiled: dict[str, list[re.Pattern]] = {}
    for cat, patterns in req.coding_scheme.items():
        compiled[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]

    # Coding matrix: 1 if text matches category, 0 otherwise
    coding_matrix = np.zeros((n_texts, n_cats), dtype=int)
    example_matches: dict[str, list[dict]] = {cat: [] for cat in categories}

    for ti, text in enumerate(req.texts):
        for ci, cat in enumerate(categories):
            for pat in compiled[cat]:
                match = pat.search(text)
                if match:
                    coding_matrix[ti, ci] = 1
                    if len(example_matches[cat]) < 5:
                        example_matches[cat].append({
                            "text_index": ti,
                            "matched": match.group(),
                            "context": text[max(0, match.start() - 30):match.end() + 30],
                        })
                    break  # one match per category per text is enough

    # Category frequencies
    cat_freqs = {
        cat: {"count": int(coding_matrix[:, ci].sum()),
              "pct": _safe_float(100 * coding_matrix[:, ci].sum() / n_texts)}
        for ci, cat in enumerate(categories)
    }

    # Co-occurrence matrix (category x category)
    co_occurrence = np.zeros((n_cats, n_cats), dtype=int)
    for i in range(n_cats):
        for j in range(n_cats):
            co_occurrence[i, j] = int(np.sum(coding_matrix[:, i] & coding_matrix[:, j]))

    co_occurrence_dict = {
        categories[i]: {
            categories[j]: int(co_occurrence[i, j])
            for j in range(n_cats)
        }
        for i in range(n_cats)
    }

    # Chi-square test for category independence (on co-occurrence)
    chi2_results: dict[str, dict] = {}
    if n_cats >= 2 and n_texts >= 5:
        for i in range(n_cats):
            for j in range(i + 1, n_cats):
                # 2x2 contingency table
                a = int(np.sum(coding_matrix[:, i] & coding_matrix[:, j]))
                b = int(np.sum(coding_matrix[:, i] & (1 - coding_matrix[:, j])))
                c = int(np.sum((1 - coding_matrix[:, i]) & coding_matrix[:, j]))
                d = int(np.sum((1 - coding_matrix[:, i]) & (1 - coding_matrix[:, j])))
                table = np.array([[a, b], [c, d]])
                try:
                    chi2, p, dof, expected = sp_stats.chi2_contingency(table, correction=True)
                    key = f"{categories[i]} × {categories[j]}"
                    chi2_results[key] = {
                        "chi2": _safe_float(chi2),
                        "p_value": _safe_float(p),
                        "dof": int(dof),
                        "contingency_table": {"a": a, "b": b, "c": c, "d": d},
                    }
                except Exception:
                    pass

    # Inter-category agreement (average pairwise kappa-like measure)
    agreement_scores: list[float] = []
    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            agree = int(np.sum(coding_matrix[:, i] == coding_matrix[:, j]))
            agreement_scores.append(agree / n_texts)
    avg_agreement = _safe_float(np.mean(agreement_scores)) if agreement_scores else 0.0

    return {
        "coding_matrix": coding_matrix.tolist(),
        "categories": categories,
        "category_frequencies": cat_freqs,
        "co_occurrence": co_occurrence_dict,
        "example_matches": example_matches,
        "chi_square_tests": chi2_results,
        "inter_category_agreement": avg_agreement,
    }
