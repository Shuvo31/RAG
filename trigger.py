from pathlib import Path
import json
import os
import sys
import time
import subprocess
import tempfile
import hashlib
from typing import Dict, Optional, List, Tuple

STATE_FILE = ".ingest_state.json"  # persisted map of {uri: {"etag": str, "sha256": str}}
EXTENSIONS = (".pdf", ".docx", ".csv", ".xlsx")


def _load_state(state_path: Path) -> Dict[str, Dict[str, str]]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state_path: Path, state: Dict[str, Dict[str, str]]) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _sha256_file(p: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _run_embeddings_script(file_path: Path, extra_args: Optional[List[str]] = None) -> Tuple[int, str]:
    """
    Calls your incremental embedder:
      python create_embeddings.py --file "<file_path>"
    """
    cmd = [sys.executable, "create_embeddings.py", "--file", str(file_path)]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + (proc.stderr or ""))
    return proc.returncode, out.strip()


# ----------------------------
# Mode A: Watch a LOCAL folder
# ----------------------------

def scan_local_folder_once(
    folder: Path,
    state_path: Path,
    pattern_exts: Tuple[str, ...] = EXTENSIONS,
    reprocess_on_hash_change: bool = True,
    embeddings_args: Optional[List[str]] = None,
) -> None:
    """
    One-shot scan of a local folder; process new/changed files.
    Call this in a loop for "watch" semantics.
    """
    state = _load_state(state_path)
    folder.mkdir(parents=True, exist_ok=True)

    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in pattern_exts:
            continue

        key = f"file://{p.resolve()}"
        prev = state.get(key, {})
        # Use mtime as "etag" surrogate for local files
        etag = str(int(p.stat().st_mtime))

        should_process = False
        if not prev:
            should_process = True
        elif prev.get("etag") != etag and reprocess_on_hash_change:
            # On mtime change, confirm content change via hash
            new_sha = _sha256_file(p)
            if new_sha != prev.get("sha256"):
                should_process = True

        if should_process:
            code, logs = _run_embeddings_script(p, embeddings_args)
            print(f"[INGEST] {p.name} -> exit={code}")
            if logs:
                print(logs)
            if code == 0:
                # Update state (store sha256 to avoid reprocessing if unchanged)
                state[key] = {"etag": etag, "sha256": _sha256_file(p)}
                _save_state(state_path, state)


def watch_local_folder(
    folder: str,
    interval_sec: int = 60,
    embeddings_args: Optional[List[str]] = None,
) -> None:
    """
    Lightweight "watch" loop without external dependencies.
    Scans the folder every `interval_sec` seconds.
    """
    folder_path = Path(folder)
    state_path = folder_path / STATE_FILE
    print(f"🔎 Watching local folder: {folder_path.resolve()} (every {interval_sec}s)")
    while True:
        try:
            scan_local_folder_once(
                folder=folder_path,
                state_path=state_path,
                embeddings_args=embeddings_args,
            )
        except KeyboardInterrupt:
            print("🛑 Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(interval_sec)


# -----------------------------------
# Mode B: Poll an AZURE BLOB container
# -----------------------------------

def poll_azure_container(
    connection_string: str,
    container_name: str,
    prefix: str = "",
    interval_sec: int = 300,
    temp_dir: Optional[str] = None,
    embeddings_args: Optional[List[str]] = None,
) -> None:
    """
    Polls an Azure Blob Storage container for new/changed files and processes them.

    Requires `azure-storage-blob` in your environment.
    """
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception as e:
        print("Please install azure-storage-blob: pip install azure-storage-blob")
        raise e

    bsc = BlobServiceClient.from_connection_string(connection_string)
    container = bsc.get_container_client(container_name)

    state_path = Path(STATE_FILE)  # Azure mode keeps state in CWD
    state = _load_state(state_path)

    _tmp_root = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
    _tmp_root.mkdir(parents=True, exist_ok=True)

    print(
        f"☁️  Polling Azure container '{container_name}' "
        f"(prefix='{prefix}') every {interval_sec}s"
    )
    while True:
        try:
            blobs = container.list_blobs(name_starts_with=prefix)
            for blob in blobs:
                name = blob.name
                if not name.lower().endswith(EXTENSIONS):
                    continue

                key = f"azure://{container_name}/{name}"
                # Prefer etag if available for efficient change detection
                etag = getattr(blob, "etag", None) or getattr(blob, "properties", {}).get("etag", None) or ""
                prev = state.get(key, {})

                should_download = False
                if not prev:
                    should_download = True
                elif etag and prev.get("etag") != etag:
                    should_download = True

                if should_download:
                    tmp_pdf = _tmp_root / f"__ingest__{os.path.basename(name).replace('/', '_')}"
                    with open(tmp_pdf, "wb") as f:
                        downloader = container.download_blob(name)
                        f.write(downloader.readall())

                    new_sha = _sha256_file(tmp_pdf)
                    if prev and prev.get("sha256") == new_sha:
                        # Content didn't actually change; update etag and continue
                        state[key] = {"etag": etag or prev.get("etag", ""), "sha256": new_sha}
                        _save_state(state_path, state)
                        continue

                    code, logs = _run_embeddings_script(tmp_pdf, embeddings_args)
                    print(f"[INGEST] {name} -> exit={code}")
                    if logs:
                        print(logs)
                    if code == 0:
                        state[key] = {"etag": etag, "sha256": new_sha}
                        _save_state(state_path, state)
        except KeyboardInterrupt:
            print("🛑 Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(interval_sec)


# -----------------
# CLI entry point
# -----------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trigger for data ingestion & embeddings generation.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Local watch mode
    p_local = sub.add_parser("local", help="Watch a local folder for new/changed files.")
    p_local.add_argument("--folder", required=True, help="Path to the local folder to watch.")
    p_local.add_argument("--interval", type=int, default=60, help="Scan interval in seconds (default: 60).")
    p_local.add_argument("--embed-args", nargs="*", default=None, help="Extra args for create_embeddings.py")

    # Azure poll mode
    p_az = sub.add_parser("azure", help="Poll an Azure Blob Storage container for new/changed files.")
    p_az.add_argument("--conn", required=True, help="Azure Storage connection string.")
    p_az.add_argument("--container", required=True, help="Container name.")
    p_az.add_argument("--prefix", default="", help="Optional path prefix within the container.")
    p_az.add_argument("--interval", type=int, default=300, help="Polling interval in seconds (default: 300).")
    p_az.add_argument("--temp-dir", default=None, help="Optional temp dir for downloads.")
    p_az.add_argument("--embed-args", nargs="*", default=None, help="Extra args for create_embeddings.py")

    args = parser.parse_args()

    if args.mode == "local":
        watch_local_folder(folder=args.folder, interval_sec=args.interval, embeddings_args=args.embed_args)
    elif args.mode == "azure":
        poll_azure_container(
            connection_string=args.conn,
            container_name=args.container,
            prefix=args.prefix,
            interval_sec=args.interval,
            temp_dir=args.temp_dir,
            embeddings_args=args.embed_args,
        )
    else:
        parser.error("Unknown mode")


if __name__ == "__main__":
    main()
