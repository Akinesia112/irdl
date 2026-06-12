"""Regenerate the packaged SOFACoustics hash registry for supported files."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

PROVIDER_ROOT = "https://sofacoustics.org/data/database"
OUTPUT_PATH = Path("src/irdl/registry/sofacoustics_hashes.json")
SUPPORTED_FILES = {
    "hutubs": [f"pp{subject}_HRIRs_{kind}.sofa" for subject in range(1, 97) for kind in ("measured", "simulated")]
}


def _hash_static_file(provider: str, filename: str) -> tuple[str, str]:
    path_key = f"{provider}/{filename}"
    request = Request(  # noqa: S310
        f"{PROVIDER_ROOT}/{path_key}",
        headers={"User-Agent": "irdl-hash-generator"},
    )
    digest = sha256()
    with urlopen(request, timeout=120) as response:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return path_key, f"sha256:{digest.hexdigest()}"


def main() -> None:
    """Regenerate the flat provider-relative-path hash registry."""
    jobs = [(provider, filename) for provider, files in SUPPORTED_FILES.items() for filename in files]
    registry: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_hash_static_file, provider, filename): (provider, filename) for provider, filename in jobs
        }
        for future in as_completed(futures):
            path_key, digest = future.result()
            registry[path_key] = digest
            print(f"hashed {path_key}")  # noqa: T201

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dict(sorted(registry.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")  # noqa: T201


if __name__ == "__main__":
    main()
