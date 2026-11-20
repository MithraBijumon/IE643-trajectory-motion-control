"""
stream_got10k.py
Streams the 69 GB xche32/got10k tar.gz from Hugging Face WITHOUT saving it. Test clips (no public
annotations) are skipped as they go past; every train clip is converted on the fly with
preprocess_got10k.process() and its raw frames deleted. Only the small processed clips stay on disk.

Every `--val_every`-th kept clip goes to val/, the rest to train/.

python scripts/stream_got10k.py --out_dir data/dataset_got10k --max_clips 300
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import requests
from huggingface_hub import get_token

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_got10k import add_filter_args, process  # noqa: E402

URL = "https://huggingface.co/datasets/xche32/got10k/resolve/main/got10k.tar.gz"


SIZE = 68934605250   # bytes of got10k.tar.gz (from the repo listing)


class ParallelReader:
    """File-like object that downloads the file in CHUNK-sized byte ranges on several threads at once
    and hands the bytes out in order, so the single gzip/tar decoder sees one sequential stream."""

    def __init__(self, url, headers, size, workers=8, chunk=4 << 20, retries=6):
        self.url, self.headers, self.size = url, headers, size
        self.chunk, self.retries, self.workers = chunk, retries, workers
        self.pool = ThreadPoolExecutor(workers)
        self.pending = deque()
        self.next_off = 0
        self.cur, self.pos, self.n = b"", 0, 0

    def _fetch(self, off):
        end = min(off + self.chunk, self.size) - 1
        for attempt in range(self.retries):
            try:
                h = dict(self.headers, Range=f"bytes={off}-{end}")
                r = requests.get(self.url, headers=h, timeout=60)   # auth header is dropped on the CDN redirect
                if r.status_code == 206 and len(r.content) == end - off + 1:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise IOError(f"range {off}-{end} failed after {self.retries} tries")

    def _next_chunk(self):
        while len(self.pending) < self.workers * 2 and self.next_off < self.size:
            self.pending.append(self.pool.submit(self._fetch, self.next_off))
            self.next_off += self.chunk
        return self.pending.popleft().result() if self.pending else b""

    def read(self, k=-1):
        if k is None or k < 0:
            k = 1 << 30
        out = []
        while k > 0:
            if self.pos >= len(self.cur):
                self.cur, self.pos = self._next_chunk(), 0
                if not self.cur:
                    break
            piece = self.cur[self.pos:self.pos + k]
            self.pos += len(piece)
            k -= len(piece)
            out.append(piece)
        data = b"".join(out)
        self.n += len(data)
        return data

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def stream(args, done):
    kept = len(done)
    headers = {"User-Agent": "got10k-stream"}
    token = get_token()                      # requests drops this header on the cross-host CDN redirect
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"authenticated: {bool(token)}, {args.workers} parallel connections", flush=True)
    src = ParallelReader(URL, headers, SIZE, workers=args.workers)
    tar = tarfile.open(fileobj=src, mode="r|gz")
    tmp = tempfile.mkdtemp(prefix="got10k_raw_")
    cur, cur_dir, last_log, skipped_test = None, None, 0, 0

    def finish():
        nonlocal kept, cur, cur_dir
        if cur is None:
            return
        split = "val" if (kept + 1) % args.val_every == 0 else "train"
        reason = process(cur_dir, os.path.join(args.out_dir, split, cur), args)
        if reason:
            print(f"[skip] {cur}: {reason}", flush=True)
        else:
            kept += 1
            done.add(cur)
            print(f"[ok]   {cur} -> {split}   (kept {kept})", flush=True)
        shutil.rmtree(cur_dir, ignore_errors=True)
        cur = cur_dir = None

    try:
        for m in tar:
            parts = m.name.split("/")
            if len(parts) < 3 or not m.isfile():
                continue
            split, clip, fn = parts[1], parts[2], parts[-1]
            if time.time() - last_log > 30:
                last_log = time.time()
                print(f"  ... {src.n / 1e6:.0f} MB streamed, in '{split}', {skipped_test} test files dropped", flush=True)
            if split == "test":
                skipped_test += 1
                continue                      # never written to disk
            if split not in ("train", "val") or clip in done:
                continue
            if clip != cur:
                finish()                      # the previous clip is complete once a new one starts
                if args.max_clips and kept >= args.max_clips:
                    break                     # stop here, before starting a clip we would not finish
                cur, cur_dir = clip, os.path.join(tmp, clip)
                os.makedirs(cur_dir)
            if fn.endswith((".jpg", ".txt", ".label")):
                with open(os.path.join(cur_dir, fn), "wb") as out:
                    shutil.copyfileobj(tar.extractfile(m), out)
        else:
            finish()                          # stream ended normally: convert the last clip
        cur = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        src.close()
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/dataset_got10k")
    p.add_argument("--val_every", type=int, default=15, help="every Nth kept clip becomes val")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=8, help="parallel range downloads")
    add_filter_args(p)
    args = p.parse_args()

    done = set()
    for attempt in range(args.retries + 1):
        try:
            kept = stream(args, done)
            print(f"\nDone: kept {kept} clips in {args.out_dir}", flush=True)
            return
        except Exception as e:  # dropped connection: restart the stream, skipping finished clips
            print(f"[retry {attempt + 1}] {type(e).__name__}: {e}", flush=True)
    print("gave up; clips kept so far are still in", args.out_dir)


if __name__ == "__main__":
    main()
