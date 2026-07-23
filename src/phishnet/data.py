"""Synthetic phishing/normal email generator and tokenizer.

The dataset used throughout this project is entirely synthetic: emails are
built by filling randomized templates with randomized fillers. No public
phishing corpus is bundled or downloaded, so results here demonstrate the
modeling approach on a controlled, reproducible dataset rather than a
real-world benchmark. See ``data/README.md`` for the rationale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# Phishing templates span several common social-engineering angles: fake
# prizes, account threats, payment pressure, tax scares, tech-support scams,
# and credential harvesting, each with independent filler pools so the
# surface form varies a lot even though the underlying tactic repeats.
PHISHING_TEMPLATES: list[str] = [
    "selamat anda menang undian {brand}! klik {url} sebelum {batas}",
    "akun {brand} anda terblokir, verifikasi segera di {url} sebelum {batas}",
    "pembayaran {brand} gagal, bayar ulang sekarang di {url}",
    "pajak anda belum dibayar, selesaikan di {url} sebelum {batas}",
    "bonus pulsa {brand} menunggu, klaim sekarang di {url}",
    "peringatan keamanan: aktivitas mencurigakan di akun {brand}, konfirmasi di {url}",
    "tim support {brand} butuh akses jarak jauh, hubungi via {url}",
    "invoice {brand} #{kode} jatuh tempo, bayar segera di {url}",
    "paket anda tertahan di bea cukai, bayar biaya di {url} sebelum {batas}",
    "update password {brand} anda kadaluarsa, ganti sekarang di {url}",
    "verifikasi ulang kartu {brand} anda di {url}, akun akan ditutup {batas}",
    "klaim hadiah {brand} #{kode} anda sebelum {batas} di {url}",
]

NORMAL_TEMPLATES: list[str] = [
    "jadwal kuliah {jurusan} minggu ini jam {jam} sudah diperbarui",
    "tagihan listrik bulan {bulan} kantor {nama} sudah dibayar otomatis",
    "reminder meeting project {nama} besok jam {jam}",
    "update status pengiriman {barang} sedang menuju {kota}",
    "konfirmasi pendaftaran {acara} di {kota} berhasil diterima",
    "notulen rapat divisi {jurusan} bulan {bulan} sudah diunggah ke drive",
    "slip gaji bulan {bulan} tim {nama} sudah bisa diunduh",
    "pengingat: {acara} akan dimulai jam {jam}",
    "laporan mingguan {nama} minggu ke-{minggu} sudah dikirim ke tim",
    "terima kasih sudah menghadiri {acara} di {kota} minggu lalu",
    "pesanan {barang} anda sudah tiba di {kota}",
    "hasil ujian {jurusan} semester ini sudah bisa dicek jam {jam}",
]

_BRAND = [
    "tokopedia", "shopee", "gojek", "dana", "ovo", "bca", "bri", "telkomsel",
    "mandiri", "bni", "grab", "linkaja",
]
_URL = [
    "bit.ly/win22", "klik.me/vrf21", "pay.me/now", "web.co/pay", "acc.io/verify",
    "go.link/claim", "sec.re/login", "id.co/reset", "conf.ly/otp", "cek.id/status",
]
_BATAS = ["24 jam", "hari ini", "besok pagi", "3 jam lagi", "1x24 jam", "akhir minggu ini"]
_KODE = [f"{n:05d}" for n in range(10000, 99999, 37)]
_JURUSAN = [
    "teknik elektro", "informatika", "mesin", "sipil", "elektronika",
    "sistem informasi", "arsitektur", "industri", "kimia", "fisika",
]
_BULAN = [
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]
_NAMA = [
    "website", "mobile app", "sistem erp", "database", "dashboard",
    "gudang", "keuangan", "produksi", "logistik", "helpdesk",
]
_BARANG = ["laptop", "hp", "monitor", "printer", "router", "keyboard", "webcam", "server"]
_ACARA = [
    "workshop ai", "seminar tech", "bootcamp coding", "hackathon", "rapat divisi",
    "pelatihan keamanan", "sesi onboarding", "town hall", "demo produk", "review sprint",
]
_JAM = ["08.00", "09.00", "10.30", "13.00", "15.00", "16.30", "19.00", "20.00"]
_KOTA = [
    "jakarta", "bandung", "surabaya", "medan", "semarang",
    "yogyakarta", "makassar", "denpasar", "malang", "palembang",
]
_MINGGU = ["1", "2", "3", "4"]


@dataclass(frozen=True)
class Dataset:
    emails: list[str]
    labels: list[int]  # 1 = phishing, 0 = normal


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        brand=rng.choice(_BRAND),
        url=rng.choice(_URL),
        batas=rng.choice(_BATAS),
        kode=rng.choice(_KODE),
        jurusan=rng.choice(_JURUSAN),
        bulan=rng.choice(_BULAN),
        nama=rng.choice(_NAMA),
        barang=rng.choice(_BARANG),
        acara=rng.choice(_ACARA),
        jam=rng.choice(_JAM),
        kota=rng.choice(_KOTA),
        minggu=rng.choice(_MINGGU),
    )


def _unique_samples(
    templates: list[str], label: int, count: int, rng: random.Random
) -> list[tuple[str, int]]:
    """Fill templates until ``count`` distinct email strings are collected.

    Deduplicating at generation time (rather than after) is what guarantees
    the train/val/test split never leaks an identical email across splits.
    """
    seen: set[str] = set()
    samples: list[tuple[str, int]] = []
    max_attempts = count * 50
    attempts = 0
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        email = _fill(rng.choice(templates), rng)
        if email in seen:
            continue
        seen.add(email)
        samples.append((email, label))
    if len(samples) < count:
        raise ValueError(
            f"could not generate {count} unique samples for label {label} "
            f"(only found {len(samples)}); reduce n_samples or add more filler variety"
        )
    return samples


def generate_dataset(n_samples: int = 600, phishing_ratio: float = 0.35, seed: int = 42) -> Dataset:
    """Generate a synthetic, labeled email dataset with no duplicate emails.

    ``phishing_ratio`` defaults to 0.35 rather than 0.5 to keep the class
    imbalance in the same ballpark as real phishing-detection datasets,
    where phishing is the minority class.
    """
    rng = random.Random(seed)  # nosec B311: synthetic data generation, not security-sensitive
    n_phishing = int(round(n_samples * phishing_ratio))
    n_normal = n_samples - n_phishing

    combined = _unique_samples(PHISHING_TEMPLATES, 1, n_phishing, rng)
    combined += _unique_samples(NORMAL_TEMPLATES, 0, n_normal, rng)
    rng.shuffle(combined)

    emails = [email for email, _ in combined]
    labels = [label for _, label in combined]
    return Dataset(emails=emails, labels=labels)


def train_val_test_split(
    dataset: Dataset, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    """Stratified split into train/val/test with no overlap."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(dataset.emails))
    train_idx, temp_idx = train_test_split(
        idx, test_size=val_size + test_size, random_state=seed, stratify=dataset.labels
    )
    temp_labels = [dataset.labels[i] for i in temp_idx]
    rel_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test_size, random_state=seed, stratify=temp_labels
    )

    def subset(indices: np.ndarray) -> Dataset:
        return Dataset(
            emails=[dataset.emails[i] for i in indices],
            labels=[dataset.labels[i] for i in indices],
        )

    return subset(train_idx), subset(val_idx), subset(test_idx)


class Vocabulary:
    """Word-level vocabulary fit only on training text to avoid leakage."""

    def __init__(self, max_len: int = 12, max_vocab: int = 200) -> None:
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.word_to_idx: dict[str, int] = {}

    def fit(self, texts: list[str]) -> Vocabulary:
        counts: dict[str, int] = {}
        for text in texts:
            for word in self._normalize(text).split():
                counts[word] = counts.get(word, 0) + 1
        top_words = sorted(counts, key=lambda w: (-counts[w], w))[: self.max_vocab]
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(top_words)}  # 0 = padding
        return self

    @property
    def size(self) -> int:
        return len(self.word_to_idx) + 1  # + padding index

    @staticmethod
    def _normalize(text: str) -> str:
        import re

        return re.sub(r"[^\w\s]", "", text.lower())

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = self._normalize(text).split()[: self.max_len]
            for j, token in enumerate(tokens):
                out[i, j] = self.word_to_idx.get(token, 0)
        return out

    def tokens(self, text: str) -> list[str]:
        return self._normalize(text).split()[: self.max_len]
