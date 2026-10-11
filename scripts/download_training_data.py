"""Retrieve the pinned public Sparkov portion and verify both archive and CSV."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

import requests

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path.home()/".cache"/"fraudshield")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Keep the full raw data outside the application repository.")
    manifest=json.loads((ROOT/"shield_artifacts"/"source_manifest.json").read_text())
    target=args.output/"sparkov_verified.csv"
    if target.exists() and digest(target)==manifest["verified_csv_sha256"]:
        print(f"Verified existing source: {target}");return
    source=manifest["source"]
    url=f"https://raw.githubusercontent.com/kashish7-7/boosting/{source['commit']}/{source['path']}"
    archive=args.output/"sparkov_verified.zip"
    part=args.output/"sparkov_verified.zip.part"
    with requests.get(url,stream=True,timeout=(25,120)) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(1024*1024):handle.write(chunk)
    if digest(part)!=source["archive_sha256"]:raise ValueError("Archive checksum mismatch; do not use this download.")
    part.replace(archive)
    with zipfile.ZipFile(archive) as contents:
        if contents.testzip() is not None:raise ValueError("Archive CRC validation failed")
        member=source["archive_member"]
        with contents.open(member) as incoming,target.open("wb") as outgoing:shutil.copyfileobj(incoming,outgoing)
    if digest(target)!=manifest["verified_csv_sha256"]:raise ValueError("CSV checksum mismatch; do not train on this file.")
    print(f"Verified training source: {target}")


if __name__=="__main__":main()
