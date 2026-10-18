"""Pure text helpers of the BI dashboard: tokens, name masking, weighted log-odds, KWIC and length bins."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import math
import re

import pandas as pd

from temajelajah.bi_data import (SENTIMENTS, fmt_int, fmt_pct, md_escape, venue_name, venue_rank, wilson_interval,  # md_escape: re-exported
                                 word_count, session_comparison_blocker)

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
NAME_TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]{3,}|[\u0400-\u04FF]{3,}")  # Latin or Cyrillic only, so "Gedeの…" splits at the script change
SENTENCE_START = frozenset(".!?…\"'“(:-")
SENTENCE_END = frozenset(".!?…")  # rare-name rule: "(Jonny & …" or "-Paige" are not sentence starts
NAME_FOLLOWERS = frozenset("""is was has had and did does will would can could were are he she made makes gave gives showed shows taught teaches
helped helps explained explains welcomed welcomes took takes also really always picked met drove cooked prepared guided guides told tells
speaks spoke knows knew who said says let lets got gets brought brings walked led leads kept keeps patiently kindly even
molto muy très sehr""".split())  # "very" without a verb: "Ketut muy amable", "… unica. Qatar molto gentile"
STRONG_FOLLOWERS = NAME_FOLLOWERS - {"and"}  # "Mandala and…" is a list; "Cathy who…" or "Liz was…" names someone
CAMEL_RE = re.compile(r"[A-Z][a-z]+[A-Z][A-Za-z]*")  # "BitAdventure": handles and user names
FOLLOWER_RE = re.compile(r"\s*(,\s*)?(\(|&|[^\W\d_]+)")
STOPWORDS_EN = frozenset("""a about above after again against all am an and any are as at be because been before being below between both
by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if
in into is it its itself just me more most my myself nor of off on once or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself yourselves also us one get got went go even really much many lot
lots well made make day time way thing things back""".split())
KEEP = frozenset({"but", "not", "no", "only", "wish", "however", "unfortunately", "didn't", "wasn't"})
ALLOW_PROPER = frozenset("""bali ubud yogyakarta jogja yogya java javanese indonesia indonesian lombok kotagede kota gede denpasar jakarta
keliki batik english airbnb balinese idr rupiah dutch german french australia australian canada usa europe asia god christmas covid monday
tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november
december mondays holidays juli wayang gamelan songket tapis ikat kebaya sarong canting tjanting pendopo kampung rindik ganesh ganesha
barong kecak legong ramayana muslim hindu buddhist islam nyepi galungan canggu sanur kuta seminyak uluwatu tegallalang kintamani gianyar
sukawati celuk mataram rinjani pringgasela senggigi gili nusa penida tanah abang petojo blok juanda aceh ampat sumatra sulawesi borneo
flores komodo papua surabaya bandung solo surakarta semarang pekalongan malioboro prambanan borobudur kraton keraton imogiri bantul
jogjakarta yogjakarta kotegede kottagede kote baliness baltik bakit batic batiks batiken popiler popilar popilier poplier widaya widja wydia
tekstil museum jak jaklingko lingko indo indonesians indonésie chinese japanese korean thai malay singapore malaysia thailand japan china
korea taiwan vietnam india america american england british london paris spain spanish italy italian germany netherlands ireland scotland
switzerland swiss austria belgium sweden norway denmark poland russia mexico brazil zealand east west north south yorker instagram
whatsapp facebook google tripadvisor klook apple nike unesco tshirt programm sesame piano fire extinguisher matur melali mahalo aloha kudos
transjakarta pak bu ibu bapak mba mbak mbok mas bli abang mister studio studios батик бали убуд убуда джокьякарта индонезия""".split())
NAME_STOPLIST = frozenset({"wayan", "nyoman", "ketut", "putu", "kadek", "komang", "gusti", "nunung", "widya", "alwan", "seno", "endah",
                           "agung", "ayu", "dewa", "dewi", "ngurah", "nengah", "nunu", "nurung", "nuning", "nanung",
                           # the same names in Cyrillic (Russian reviews); matched from a word start, so declined forms are masked too
                           "вайан", "ваян", "ньоман", "кетут", "путу", "кадек", "команг", "густи", "агунг", "нгурах", "ненгах", "нунунг",
                           "видья", "алван"})
VENUE_TOKENS = frozenset({"nunung", "popiler", "widya", "alwan", "five", "arts", "art", "studio", "seno", "giriloyo", "museum", "tekstil"})
MASK = "[nama]"
HONORIFIC_RE = re.compile(r"(?<![A-Za-z])(?:[Pp]ak|[Bb]u|[Ii]bu|[Bb]apak|[Mm]bak|[Mm]ba|[Mm]bok|[Bb]li|Mas|Mrs|Mr|Ms|Miss|Mister|Herr|Frau|"
                          r"Madame|Mme|Monsieur|Señora|Señor)\.?[ \t]+([A-Z][A-Za-z\u00C0-\u024F]*(?:['’]s)?)")
CONTACT_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<![\w.@])@[A-Za-z0-9_.]{2,}")  # e-mail addresses and @handles
NAME_WORD = r"[A-Z\u00C0-\u00DE][a-z\u00DF-\u00FF\u0100-\u024F'’]+"
JOIN = r"\s*(?:,|&|\+|\band\b|\bund\b|\bet\b|\ben\b|\by\b|\be\b)\s*"
NAMES = rf"{NAME_WORD}(?:{JOIN}{NAME_WORD})*"
TAIL = r"[^\w]*$"
GREETINGS = (r"best|regards|kind regards|cheers|greetings|liebe grüße|viele grüße|beste grüße|grüße|gruß|gruss|groetjes|groeten|bisous|bises|"
             r"saludos|abrazos|xoxo|xx")
SIGNOFF_RES = (re.compile(rf"(?:^|\s)[-–—~]+\s*({NAMES}){TAIL}"),  # "-Paige and Alex"
               re.compile(rf"(?:^|[.!?…,\n])\s*(?i:{GREETINGS})\b[\s,!.:]*({NAMES}){TAIL}"),  # "Best, Sophie", "liebe Grüße Luisa :)"
               re.compile(rf"[.!?…]\s+({NAME_WORD}(?:{JOIN}{NAME_WORD})+){TAIL}"),  # "Thanks a lot! Carolin and Lisa"
               re.compile(rf"\n\s*({NAME_WORD}){TAIL}"))  # a name alone on the last line
CLOSERS = frozenset("""thanks thank thankyou thanx thx terimakasih terima kasih makasih merci danke gracias grazie bedankt cheers aloha ciao
bye love best regards greetings recommend recommended enjoy enjoyed wonderful amazing awesome great perfect fantastic fabulous excellent lovely
beautiful brilliant worthwhile highly definitely absolutely company team family friends everyone everybody all staff""".split())
READABLE = CLOSERS | ALLOW_PROPER  # sign-off words that stay visible
TERM_MODES = ("Keluhan vs lainnya", "Per venue")
TERM_COLS = ["term", "z", "df_a", "df_b", "n_a", "n_b", "docs_a", "docs_b", "sisi", "label"]
VENUE_TERM_COLS = ["venue_key", "venue", "n_docs", "term", "z", "df_a", "df_b", "n_a", "n_b", "docs_a", "docs_b", "label"]
MIN_GROUP, Z_EPS = 5, 1e-9
KWIC_COLS = ["Venue", "Tahun", "★", "Sentimen", "Kiri", "Kata", "Kanan"]
CJK_QUERY = re.compile(r"[\u1100-\u11FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]")
LENGTH_BINS = ((20, "≤20"), (40, "21–40"), (80, "41–80"), (160, "81–160"), (math.inf, ">160"))


def tokenize(text: str, drop: frozenset[str] = frozenset()) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(text.casefold().replace("’", "'")):
        token = token[:-2] if token.endswith("'s") else token
        if token not in drop and (token in KEEP or (len(token) >= 3 and token not in STOPWORDS_EN)):
            tokens.add(token)
    return tokens


def _sentence_initial(text: str, start: int, marks: frozenset[str] = SENTENCE_START) -> bool:
    before = start - 1
    while before >= 0 and text[before].isspace():
        before -= 1
    return before < 0 or text[before] in marks


def _names_context(text: str, end: int) -> bool:
    """A sentence-initial word reads as a name when a verb, 'and', '&' or a parenthesis follows it ("Dik was…", "Anton and I…")."""
    follower = FOLLOWER_RE.match(text, end)
    if follower is None:
        return False
    token = follower.group(2)
    return token in ("(", "who") if follower.group(1) else token in ("(", "&") or token.casefold() in NAME_FOLLOWERS


def _strong_name_context(text: str, end: int) -> bool:
    """A verb or ', who' right after the word: "her sister Cathy who…", "Liz was…"."""
    follower = FOLLOWER_RE.match(text, end)
    if follower is None:
        return False
    token = follower.group(2).casefold()
    return token == "who" if follower.group(1) else token in STRONG_FOLLOWERS


def proper_noun_vocab(texts: Iterable[str], min_count: int = 3, share: float = 0.8) -> frozenset[str]:
    """Names: tokens capitalised in >=80% of >=3 mid-sentence uses (spec M2), plus title-case tokens never written in lower case
    anywhere that appear mid-sentence or open a sentence followed by a verb (rare names, misspellings, handles, sign-offs).
    An informal lower-case use ("atencion de cathy") does not clear a name that is written, at least as often, capitalised
    mid-sentence right before a verb ("her sister Cathy who…"). Known places, culture words and honorifics are kept readable."""
    mid, capital, rare_uses, strong, lower = Counter(), Counter(), Counter(), Counter(), Counter()
    for text in texts:
        for match in NAME_TOKEN_RE.finditer(text):
            word = match.group()
            key = word.casefold()
            if word.islower():
                lower[key] += 1
            titled = word[0].isupper() and not word.isupper()
            if titled and (not _sentence_initial(text, match.start(), SENTENCE_END) or CAMEL_RE.fullmatch(word)
                           or (not key.endswith("ly") and _names_context(text, match.end()))):
                rare_uses[key] += 1
            if _sentence_initial(text, match.start()):
                continue
            mid[key] += 1
            capital[key] += word[0].isupper()
            strong[key] += titled and _strong_name_context(text, match.end())
    frequent = {word for word, n in mid.items() if n >= min_count and capital[word] / n >= share}
    rare = {word for word in rare_uses if lower[word] <= strong[word]}
    return frozenset(((frequent | rare) - ALLOW_PROPER) | NAME_STOPLIST)


def name_vocab(*frames: pd.DataFrame | None) -> frozenset[str]:
    """Names learned from every text the page can show (corpus incl. owner replies, labels, aspect labels). German is left out
    of the statistics because it capitalises every noun; its names are still masked when they occur elsewhere or in a stoplist."""
    texts: dict[str, None] = {}
    for frame in frames:
        if frame is not None:
            texts.update(dict.fromkeys(frame.loc[frame["source_locale"].ne("de"), "clean_text"]))
    return proper_noun_vocab(texts)


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def redact(text: str, spans: Sequence[tuple[int, int]], start: int = 0, end: int | None = None) -> str:
    """text[start:end] with every part covered by a span replaced by [nama]."""
    end = len(text) if end is None else end
    out, position = [], start
    for left, right in spans:
        if right <= start or left >= end:
            continue
        left, right = max(left, start), min(right, end)
        out += [text[position:left], MASK]
        position = right
    return "".join(out) + text[position:end]


class NameMasker:
    """Heuristic masking of person names in displayed snippets: learned names, the word after an honorific, e-mail addresses,
    @handles and trailing sign-offs. Callable like a function; `spans` lets KWIC mask the full review before cutting it."""

    def __init__(self, vocab: Iterable[str]):
        names = sorted({name.casefold() for name in vocab if name}, key=lambda name: (-len(name), name))
        # Letter edges instead of \b: Hangul, kana and Han count as \w, so "Wayan과" would slip through \b. A name starts at a
        # Latin or Cyrillic word start; only Latin letters close it, so Russian case endings ("Вайаном") are masked with the stem.
        self.pattern = (re.compile(r"(?<![A-Za-z\u0400-\u04FF])(" + "|".join(map(re.escape, names)) + r")(?:['’]s)?(?![A-Za-z])", re.I)
                        if names else None)

    def spans(self, text: str) -> list[tuple[int, int]]:
        found = [match.span() for match in self.pattern.finditer(text)] if self.pattern else []
        found += [match.span(1) for match in HONORIFIC_RE.finditer(text)]
        found += [match.span() for match in CONTACT_RE.finditer(text)]
        for rule in SIGNOFF_RES:
            match = rule.search(text)
            if match:
                base = match.start(1)
                found += [(base + word.start(), base + word.end()) for word in re.finditer(NAME_WORD, match.group(1))
                          if word.group().casefold().rstrip("'’") not in READABLE]
        return _merge(found)

    def __call__(self, text: str) -> str:
        return redact(text, self.spans(text))


def name_masker(vocab: Iterable[str]) -> NameMasker:
    return NameMasker(vocab)


def _masked_parts(text: str, mask: Callable[[str], str], cuts: Sequence[tuple[int, int]]) -> list[str]:
    spans = mask.spans(text) if isinstance(mask, NameMasker) else None
    return [mask(text[a:b]) if spans is None else redact(text, spans, a, b) for a, b in cuts]


def log_odds_dirichlet(docs_a: Sequence[set[str]], docs_b: Sequence[set[str]], alpha0: float = 1000.0, min_df: int = 5) -> pd.DataFrame:
    """Exploratory weighted log odds on binary document frequencies, without a significance threshold.
    n_a/n_b are SUMMED document-frequency mass over the retained vocabulary, not numbers of reviews (docs_a/docs_b).
    Tokens within a review are dependent, so z is a descriptive ranking score, not a calibrated hypothesis test.
    """
    df_a, df_b = Counter(), Counter()
    for doc in docs_a:
        df_a.update(doc)
    for doc in docs_b:
        df_b.update(doc)
    vocab = sorted(term for term in set(df_a) | set(df_b) if df_a[term] + df_b[term] >= min_df)
    pooled = sum(df_a[term] + df_b[term] for term in vocab)
    n_a, n_b = sum(df_a[term] for term in vocab), sum(df_b[term] for term in vocab)
    rows = []
    for term in vocab:
        alpha = alpha0 * (df_a[term] + df_b[term]) / pooled
        rest_a, rest_b = n_a + alpha0 - df_a[term] - alpha, n_b + alpha0 - df_b[term] - alpha
        if rest_a <= 0 or rest_b <= 0:
            continue
        delta = math.log((df_a[term] + alpha) / rest_a) - math.log((df_b[term] + alpha) / rest_b)
        rows.append((term, delta / math.sqrt(1 / (df_a[term] + alpha) + 1 / (df_b[term] + alpha)), df_a[term], df_b[term],
                     n_a, n_b, len(docs_a), len(docs_b)))
    frame = pd.DataFrame(rows, columns=["term", "z", "df_a", "df_b", "n_a", "n_b", "docs_a", "docs_b"])
    return frame.sort_values(["z", "term"], ascending=[False, True], ignore_index=True)


def _split(docs: list[set[str]], inside: Iterable[bool]) -> tuple[list[set[str]], list[set[str]]]:
    flags = list(inside)
    return [doc for doc, flag in zip(docs, flags) if flag], [doc for doc, flag in zip(docs, flags) if not flag]


def terms_blocker(labels: pd.DataFrame, mode: str, *, min_group: int = MIN_GROUP) -> str | None:
    """Why distinctive terms cannot be computed on this slice: a comparison group that is empty or tiny gives only rounding noise."""
    english = labels[labels["source_locale"].eq("en")]
    blocker = session_comparison_blocker(english)
    if blocker:
        return blocker
    if mode == TERM_MODES[0]:
        k = int(english["complaint"].sum())
        if k < min_group or len(english) - k < min_group:
            return (f"Butuh ≥{min_group} ulasan berkeluhan dan ≥{min_group} ulasan lain berbahasa Inggris untuk dibandingkan; "
                    f"pada filter ini ada {fmt_int(k)} dan {fmt_int(len(english) - k)}.")
    elif english["venue_key"].nunique() < 2:
        return ("Mode per venue membandingkan tiap venue dengan venue lain, jadi butuh ≥2 venue yang punya ulasan berbahasa Inggris "
                "pada filter ini. Tambah venue atau kosongkan filter venue.")
    return None


def distinctive_terms(labels: pd.DataFrame, mode: str, drop: frozenset[str], *, top_pos: int = 10, top_neg: int = 6,
                      per_venue: int = 8, min_docs: int = 30, min_group: int = MIN_GROUP) -> tuple[pd.DataFrame, list[str]]:
    if mode not in TERM_MODES:
        raise ValueError(f"Mode kata pembeda tidak dikenal: {mode}")
    if terms_blocker(labels, mode, min_group=min_group):
        return pd.DataFrame(columns=TERM_COLS if mode == TERM_MODES[0] else VENUE_TERM_COLS), []
    english = labels[labels["source_locale"].eq("en")]
    docs = [tokenize(text, frozenset(drop) | VENUE_TOKENS) for text in english["clean_text"]]
    if mode == TERM_MODES[0]:
        result = log_odds_dirichlet(*_split(docs, english["complaint"].astype(bool)))
        picked = pd.concat([result[result["z"] > Z_EPS].head(top_pos), result[result["z"] < -Z_EPS].tail(top_neg).iloc[::-1]], ignore_index=True)
        side = picked["z"] > 0
        picked["sisi"] = side.map({True: "ulasan berkeluhan", False: "ulasan lain"}).astype(object)
        picked["label"] = [f"{fmt_int(a if up else b)} ulasan" for a, b, up in zip(picked["df_a"], picked["df_b"], side)]
        return picked.reindex(columns=TERM_COLS), []
    counts = english["venue_key"].value_counts().reindex(labels["venue_key"].unique(), fill_value=0)  # venues without English text too
    untagged = labels.loc[labels["source_locale"].eq(""), "venue_key"].value_counts()  # no language metadata: never analysed
    frames, skipped = [], []
    for venue in sorted(counts.index, key=lambda key: (venue_rank(key), key)):
        n = int(counts[venue])
        if n < min_docs or len(english) - n < min_group:
            missing = int(untagged.get(venue, 0))
            skipped.append(f"{venue_name(venue)} ({fmt_int(n)}" + (f"; {fmt_int(missing)} tanpa metadata bahasa)" if missing else ")"))
            continue
        top = log_odds_dirichlet(*_split(docs, english["venue_key"].eq(venue)))
        top = top[top["z"] > Z_EPS].head(per_venue)
        frames.append(top.assign(venue_key=venue, venue=venue_name(venue), n_docs=n,
                                 label=[f"{fmt_int(a)} ulasan" for a in top["df_a"]]))
    terms = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=VENUE_TERM_COLS)
    return terms.reindex(columns=VENUE_TERM_COLS), skipped


def kwic(labels: pd.DataFrame, query: str, mask: Callable[[str], str], *, window: int = 70, limit: int = 30,
         only_complaints: bool = False) -> tuple[pd.DataFrame, dict]:
    query = query.strip()[:40]
    stats = {"query": query, "matches": 0, "complaints": 0, "share": math.nan, "lo": math.nan, "hi": math.nan, "shown": 0}
    if len(query) < 2:
        return pd.DataFrame(columns=KWIC_COLS), stats
    # Either apostrophe matches either, as tokenize() folds them: "didn't" from the terms chart also finds "didn’t".
    needle = re.escape(query.replace("’", "'")).replace("'", "['’]")
    # Han, kana and Hangul attach particles without spaces, so a CJK query matches as a substring; others need whole words.
    pattern = re.compile(needle if CJK_QUERY.search(query) else rf"(?<!\w){needle}(?!\w)", re.I)
    hits = labels["clean_text"].map(pattern.search)
    found = labels[hits.notna()]
    k, n = int(found["complaint"].sum()), len(found)
    stats |= {"matches": n, "complaints": k, "share": k / n if n else math.nan, **dict(zip(("lo", "hi"), wilson_interval(k, n)))}
    if only_complaints:
        found = found[found["complaint"]]
    found = found.assign(_complaint=found["complaint"].astype(int), _year=found["year"].fillna(0).astype(int))
    found = found.sort_values(["_complaint", "_year"], ascending=False, kind="stable").head(limit)
    rows = []
    for index, row in found.iterrows():
        match, text, sentiment = hits[index], row["clean_text"], row["sentiment_4class"]
        left, word, right = _masked_parts(text, mask, ((0, match.start()), match.span(), (match.end(), len(text))))
        left, right = re.sub(r"\s+", " ", left), re.sub(r"\s+", " ", right)
        rows.append({"Venue": venue_name(row["venue_key"]), "Tahun": row["year"], "★": row["rating_num"],
                     "Sentimen": f"{SENTIMENTS.get(sentiment, '·')} {sentiment}", "Kiri": "…" + left[-window:] if len(left) > window else left,
                     "Kata": word, "Kanan": right[:window] + "…" if len(right) > window else right})
    frame = pd.DataFrame(rows, columns=KWIC_COLS).astype({"Tahun": "Int64"})
    frame["★"] = pd.to_numeric(frame["★"], errors="coerce").round().astype("Int64")
    stats["shown"] = len(frame)
    return frame, stats


def length_bins(labels: pd.DataFrame) -> pd.DataFrame:
    words = labels["word_count"] if "word_count" in labels else labels["clean_text"].map(word_count)
    names = [name for _, name in LENGTH_BINS]
    groups = pd.cut(words.astype(float), [-math.inf, *(edge for edge, _ in LENGTH_BINS)], labels=names, right=True)
    rows = []
    for order, name in enumerate(names):
        inside = labels.loc[groups.eq(name).to_numpy(), "complaint"]
        n, k = len(inside), int(inside.sum())
        share = k / n if n else math.nan
        lo, hi = wilson_interval(k, n)
        rows.append({"urut": order, "bin": name, "n": n, "k": k, "share": share, "lo": lo, "hi": hi, "label": fmt_pct(share),
                     "axis": f"{name}\n(n={fmt_int(n)})"})
    return pd.DataFrame(rows)
