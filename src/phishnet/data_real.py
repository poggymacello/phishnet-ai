"""Loader for the real PhiUSIIL phishing URL dataset.

Two feature sets are distinguished, not one, because they have different
operational realism:

- ``URL_ONLY_FEATURES``: derivable from the URL string alone, before ever
  connecting to the site. This is what a real pre-click filter (browser
  extension, email gateway) could actually use.
- ``PAGE_CONTENT_FEATURES``: derived from the rendered page (line count,
  image count, title match score, ...). Real signal, but only available
  *after* fetching the page — which is also why this project's API never
  fetches a user-submitted URL server-side (a live URL-fetching endpoint
  is a straightforward SSRF vector). Reported as an upper-bound comparison,
  not deployed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "phiusiil.csv"

# URLSimilarityIndex is dropped, not just unused: it is exactly 100.0 for
# every single legitimate row (zero variance) and varies only for phishing
# rows -- essentially the label encoded as a "similarity score" rather than
# a behavioral signal. See README's Leakage Controls section.
LEAKAGE_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title", "URLSimilarityIndex"]

URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "IsHTTPS", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "CharContinuationRate", "URLCharProb", "TLDLegitimateProb",
]

PAGE_CONTENT_FEATURES = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef",
]

ALL_FEATURES = URL_ONLY_FEATURES + PAGE_CONTENT_FEATURES


@dataclass(frozen=True)
class UrlDataset:
    url_only_features: pd.DataFrame
    page_content_features: pd.DataFrame
    labels: np.ndarray
    urls: pd.Series
    etld1: pd.Series

    def __len__(self) -> int:
        return len(self.labels)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def _etld_plus_one(domain: str) -> str:
    ext = tldextract.extract(str(domain))
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def prepare(df: pd.DataFrame) -> UrlDataset:
    etld1 = df["Domain"].apply(_etld_plus_one)
    url_only = df[URL_ONLY_FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    page_content = df[PAGE_CONTENT_FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # PhiUSIIL's label convention is 1 = legitimate, 0 = phishing; this
    # project (and every other repo in this portfolio) uses 1 = the thing
    # you're trying to detect, so it's flipped here to 1 = phishing.
    labels = 1 - df["label"].to_numpy()
    return UrlDataset(
        url_only_features=url_only,
        page_content_features=page_content,
        labels=labels,
        urls=df["URL"],
        etld1=etld1,
    )


def group_split(
    dataset: UrlDataset, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> tuple[UrlDataset, UrlDataset, UrlDataset]:
    """Split by eTLD+1 so no domain's URLs appear in more than one split.

    A random row-level split would let the model see other URLs from the
    same domain during training and "recognize" it at test time -- the
    single largest realistic leakage risk in URL-based phishing detection,
    since a real deployment will constantly see brand-new domains it has
    never seen a single example of.
    """
    rng = np.random.default_rng(seed)
    unique_domains = np.array(dataset.etld1.unique().tolist(), dtype=object)
    rng.shuffle(unique_domains)

    n = len(unique_domains)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    train_domains = set(unique_domains[:train_end])
    val_domains = set(unique_domains[train_end:val_end])
    test_domains = set(unique_domains[val_end:])

    def subset(domains: set[str]) -> UrlDataset:
        mask = dataset.etld1.isin(domains).to_numpy()
        return UrlDataset(
            url_only_features=dataset.url_only_features.loc[mask].reset_index(drop=True),
            page_content_features=dataset.page_content_features.loc[mask].reset_index(drop=True),
            labels=dataset.labels[mask],
            urls=dataset.urls.loc[mask].reset_index(drop=True),
            etld1=dataset.etld1.loc[mask].reset_index(drop=True),
        )

    return subset(train_domains), subset(val_domains), subset(test_domains)


class CharVocabulary:
    """Character-level vocabulary for URL strings, fit only on training data.

    URLs don't have "words" the way the synthetic email generator's text
    does, so this tokenizes at the character level instead (matching the
    character n-gram framing used for the TF-IDF baseline on this dataset),
    rather than reusing the word-level ``Vocabulary`` built for v1.
    """

    def __init__(self, max_len: int = 100, max_vocab: int = 100) -> None:
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.char_to_idx: dict[str, int] = {}

    def fit(self, urls: pd.Series) -> CharVocabulary:
        counts: dict[str, int] = {}
        for url in urls:
            for ch in str(url).lower():
                counts[ch] = counts.get(ch, 0) + 1
        top_chars = sorted(counts, key=lambda c: (-counts[c], c))[: self.max_vocab]
        self.char_to_idx = {ch: idx + 1 for idx, ch in enumerate(top_chars)}  # 0 = padding
        return self

    @property
    def size(self) -> int:
        return len(self.char_to_idx) + 1

    def encode(self, urls: pd.Series) -> np.ndarray:
        out = np.zeros((len(urls), self.max_len), dtype=np.int64)
        for i, url in enumerate(urls):
            chars = str(url).lower()[: self.max_len]
            for j, ch in enumerate(chars):
                out[i, j] = self.char_to_idx.get(ch, 0)
        return out

    def tokens(self, url: str) -> list[str]:
        return list(str(url).lower()[: self.max_len])


SUSPICIOUS_URL_CHARS = set("@%-_=&?0123456789")


def is_suspicious_char(ch: str) -> bool:
    """Digits and common obfuscation/lookalike characters in URLs."""
    return ch in SUSPICIOUS_URL_CHARS or bool(re.match(r"[0-9]", ch))
