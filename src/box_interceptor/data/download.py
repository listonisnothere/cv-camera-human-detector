"""Скачивание файлов: проверка размера, атомарная запись, параллельность.

Многие серверы ограничивают скорость НА ОДНО СОЕДИНЕНИЕ (EPFL ~50 КБ/с,
хранилище весов RF-DETR ~80 КБ/с). Два приёма:
  1. несколько файлов — параллельно (download_many);
  2. один большой файл — кусками через HTTP Range, куски тоже параллельно
     (download с parts > 1). Сервер должен отвечать "Accept-Ranges: bytes".
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

USER_AGENT = "box-interceptor-downloader/0.1"
MIN_PART_SIZE = 8 << 20  # кусками качаем только файлы больше 8 МБ


def _head(url: str) -> tuple[int | None, bool]:
    """(размер, поддерживает ли сервер докачку по диапазонам)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # редиректы urllib проходит сам
        size = resp.headers.get("Content-Length")
        ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    return (int(size) if size else None), ranges


def md5sum(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 — не для безопасности, а для сверки с опубликованной суммой
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_range(url: str, path: Path, start: int, end: int) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp, path.open("r+b") as out:
        if resp.status != 206:
            raise OSError(f"Сервер проигнорировал Range (HTTP {resp.status})")
        out.seek(start)
        shutil.copyfileobj(resp, out, length=1 << 20)


def download(url: str, dest: Path, parts: int = 8, md5: str | None = None) -> Path:
    """Скачивает url в dest. Если файл уже есть и размер (и md5) совпадает — пропускает."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, ranges = _head(url)
    already_ok = dest.exists() and (size is None or dest.stat().st_size == size)
    if already_ok and (md5 is None or md5sum(dest) == md5):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    if size and ranges and parts > 1 and size > MIN_PART_SIZE:
        # Заранее создаём файл нужного размера: каждый кусок пишет в своё место
        with tmp.open("wb") as f:
            f.truncate(size)
        bounds = [(i * size // parts, (i + 1) * size // parts - 1) for i in range(parts)]
        with ThreadPoolExecutor(max_workers=parts) as pool:
            for fut in [pool.submit(_fetch_range, url, tmp, a, b) for a, b in bounds]:
                fut.result()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)

    if size is not None and tmp.stat().st_size != size:
        raise OSError(f"{url}: скачано {tmp.stat().st_size} байт из {size}")
    if md5 is not None and (got := md5sum(tmp)) != md5:
        tmp.unlink()
        raise OSError(f"{url}: MD5 не совпал ({got} != {md5}) — файл повреждён")
    tmp.rename(dest)  # атомарно: недокачанный файл никогда не выглядит готовым
    return dest


def download_many(jobs: Iterable[tuple[str, Path]], workers: int = 12) -> list[Path]:
    """Скачивает пары (url, путь) параллельно. Падает, если хоть один файл не скачался."""
    jobs = list(jobs)
    done: list[Path] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, url, dest): url for url, dest in jobs}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                path = fut.result()
                done.append(path)
                print(f"  ✓ {path.name}")
            except Exception as exc:  # noqa: BLE001 — собираем все ошибки, а не падаем на первой
                errors.append(f"{url}: {exc}")
                print(f"  ✗ {url}: {exc}")
    if errors:
        raise RuntimeError("Не удалось скачать:\n" + "\n".join(errors))
    return done
