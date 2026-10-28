"""Download only when explicitly executed on the cloud machine."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

DRIVE = "https://drive.google.com/drive/folders/1cV0ZTk92PguKS2nyii6dLV0IfqoSDHsQ"
LINCS2020 = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/"
LINCS2020_FILES = [
    "level3/level3_beta_ctl_n188708x12328.gctx",
    "level3/level3_beta_trt_cp_n1805898x12328.gctx",
    "instinfo_beta.txt",
    "geneinfo_beta.txt",
    "compoundinfo_beta.txt",
]
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
GEO_FILES = [
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz",
    "GSE92742_Broad_LINCS_inst_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "GSE92742_Broad_LINCS_gene_info.txt.gz",
]


def fetch(url, path, expected_sha256=None):
    """Resume HTTP downloads; publish only after a complete response."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_sha256:
            check_hash(path, expected_sha256)
        return path
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if offset and response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Server returned a mismatched partial range")
            mode = "ab"
        else:
            offset, mode = 0, "wb"
        remaining = int(response.headers.get("Content-Length", 0))
        received = 0
        with open(partial, mode) as f, tqdm(total=offset + remaining or None,
                                           initial=offset, unit="B", unit_scale=True,
                                           desc=path.name) as bar:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
                    bar.update(len(chunk))
        if remaining and received != remaining:
            raise IOError(f"Incomplete download: {path}")
    if expected_sha256:
        check_hash(partial, expected_sha256)
    partial.replace(path)
    return path


def check_hash(path, expected):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest().lower() != expected.lower():
        raise IOError(f"SHA256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmap = sub.add_parser("cmap")
    cmap.add_argument("--release", choices=["lincs2020", "gse92742"], default="lincs2020")
    cmap.add_argument("--output", default="data/raw")
    cmap.add_argument("--metadata-only", action="store_true")
    cmap.add_argument("--decompress", action="store_true")
    kg = sub.add_parser("mapkg")
    kg.add_argument("--output", default="weights/mapkg")
    kg.add_argument("--url", default=DRIVE)
    custom = sub.add_parser("manifest")
    custom.add_argument("manifest", help='JSON list of {url, filename, sha256 (optional)}')
    custom.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    if args.command == "mapkg":
        import gdown
        Path(args.output).mkdir(parents=True, exist_ok=True)
        files = gdown.download_folder(url=args.url, output=args.output, quiet=False)
        if not files:
            raise RuntimeError("Drive download failed; download via browser and supply local paths")
    elif args.command == "manifest":
        for item in json.loads(Path(args.manifest).read_text()):
            name = Path(item["filename"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Manifest filenames must be relative paths without '..'")
            fetch(item["url"], Path(args.output) / name, item.get("sha256"))
    else:
        files = LINCS2020_FILES if args.release == "lincs2020" else GEO_FILES
        base = LINCS2020 if args.release == "lincs2020" else GEO
        for name in files:
            if args.metadata_only and ".gctx" in name:
                continue
            path = fetch(base + name, Path(args.output) / Path(name).name)
            if args.decompress and name.endswith(".gctx.gz"):
                output = path.with_suffix("")
                if not output.exists():
                    temp = output.with_name(output.name + ".part")
                    with gzip.open(path, "rb") as src, open(temp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    temp.replace(output)


if __name__ == "__main__":
    main()
