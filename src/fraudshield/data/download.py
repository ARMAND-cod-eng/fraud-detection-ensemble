"""Dataset acquisition with integrity verification.

Order of preference:
1. An existing local file (verified by SHA-256).
2. The Kaggle CLI (``kaggle datasets download -d mlg-ulb/creditcardfraud``) when the user has
   configured their own Kaggle API token. Credentials are never read or stored by this code.
3. The public copy used by the official TensorFlow imbalanced-data tutorial.

All sources must produce the same file; the SHA-256 check guarantees it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from fraudshield.data.creditcard import sha256_file
from fraudshield.data.schema import EXPECTED_SHA256

PUBLIC_MIRROR = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def _verify(path: Path) -> bool:
    return path.exists() and sha256_file(path) == EXPECTED_SHA256


def _try_kaggle(dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmp],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
        archive = next(Path(tmp).glob("*.zip"), None)
        if archive is None:
            return False
        with zipfile.ZipFile(archive) as zf:
            zf.extract("creditcard.csv", dest.parent)
    return _verify(dest)


def ensure_creditcard(dest: str | Path = "data/raw/creditcard.csv", allow_public_mirror: bool = True) -> Path:
    dest = Path(dest)
    if _verify(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _try_kaggle(dest):
        return dest
    if not allow_public_mirror:
        raise RuntimeError(
            "Dataset not found. Install the Kaggle CLI and configure your API token, then run "
            f"`kaggle datasets download -d {KAGGLE_DATASET}` and unzip into {dest.parent}."
        )
    tmp = dest.with_suffix(".partial")
    urllib.request.urlretrieve(PUBLIC_MIRROR, tmp)
    if sha256_file(tmp) != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded file failed SHA-256 verification; refusing to use it")
    tmp.replace(dest)
    return dest
