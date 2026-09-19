"""Shared, dependency-light helpers for the controlled app-quality experiment."""

import base64
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "parts/traffic_light/configs/app_quality.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def new_directory(path):
    path = Path(path).resolve()
    # An existing empty directory may be an unfinished run. Never reuse it.
    path.mkdir(parents=True, exist_ok=False)
    return path


def child(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"상대 경로만 허용합니다: {relative}")
    return Path(root) / path


def dimensions(width, height, max_side=960):
    scale = min(1, max_side / max(width, height))
    # JavaScript Math.round for positive dimensions (Python round is ties-to-even).
    return max(1, math.floor(width * scale + .5)), max(1, math.floor(height * scale + .5))


def scale_crop(box, old_size, new_size):
    sx, sy = new_size[0] / old_size[0], new_size[1] / old_size[1]
    return (max(0, math.floor(box[0] * sx)), max(0, math.floor(box[1] * sy)),
            min(new_size[0], math.ceil(box[2] * sx)), min(new_size[1], math.ceil(box[3] * sy)))


# Mirrors gildongmu-test-app/backend/static/js/camera.js capture().
# Keep browser defaults for canvas smoothing and color space, just as the app does.
CANVAS_JPEG = """async ({maxSide, quality}) => {
  const input = document.querySelector('#source');
  const file = input.files[0];
  if (!file) throw new Error('Missing input file');
  const url = URL.createObjectURL(file);
  const img = new Image();
  const canvas = document.createElement('canvas');
  try {
    img.src = url; await img.decode();
    const original = [img.naturalWidth, img.naturalHeight];
    const scale = Math.min(1, maxSide / Math.max(...original));
    const w = Math.max(1, Math.round(original[0] * scale));
    const h = Math.max(1, Math.round(original[1] * scale));
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', {willReadFrequently: false});
    ctx.drawImage(img, 0, 0, w, h);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    if (!blob || blob.type !== 'image/jpeg') throw new Error('JPEG encoding failed');
    // Only the resized JPEG crosses the automation protocol, never the full source.
    const encoded = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error); reader.readAsDataURL(blob);
    });
    return {encoded, width:w, height:h, original};
  } finally {
    img.removeAttribute('src');
    canvas.width = 0; canvas.height = 0;
    URL.revokeObjectURL(url);
    input.value = '';
  }
}"""


class BrowserJPEG:
    def __init__(self, settings, restart_every=50):
        self.settings = settings
        if restart_every < 1:
            raise ValueError("restart_every는 1 이상이어야 합니다.")
        self.restart_every = restart_every
        self.browser = None
        self.processed = 0

    def _start_browser(self):
        self.browser = self.manager.chromium.launch(channel="chromium", headless=True)
        self.page = self.browser.new_page()
        self.page.set_content('<input type="file" id="source">')

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self.manager = sync_playwright().start()
        try:
            self._start_browser()
            self.metadata = {"engine": "chromium", "version": self.browser.version,
                             "playwright": version("playwright"), **self.settings,
                             "input_transfer": "local_file", "browser_restart_every": self.restart_every,
                             "script_sha256": hashlib.sha256(CANVAS_JPEG.encode()).hexdigest()}
            return self
        except BaseException:
            self.manager.stop()
            raise

    def encode(self, source):
        from PIL import Image
        # Labels in the existing manifests are in stored pixel coordinates.
        # Reject orientation ambiguity rather than silently rotating only the pixels.
        with Image.open(source) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"EXIF orientation과 라벨 좌표 확인이 필요합니다: {source}")
            original = image.size
        if self.processed and self.processed % self.restart_every == 0:
            # Recycle the renderer/native caches AND Playwright's driver bookkeeping.
            # A page reload alone does not bound browser-process memory over large runs.
            self.browser.close()
            self.manager.stop()
            from playwright.sync_api import sync_playwright
            self.manager = sync_playwright().start()
            self._start_browser()
        self.page.locator('#source').set_input_files(str(Path(source).resolve()))
        result = self.page.evaluate(CANVAS_JPEG, {"maxSide": self.settings["max_side"],
                                                "quality": self.settings["quality"]})
        self.processed += 1
        if tuple(result["original"]) != original:
            raise ValueError(f"브라우저 이미지 크기가 라벨 기준과 다릅니다: {source}")
        size = (result["width"], result["height"])
        if size != dimensions(*original, self.settings["max_side"]):
            raise ValueError("예상하지 못한 canvas 크기")
        return base64.b64decode(result["encoded"].split(",", 1)[1]), original, size

    def __exit__(self, *exc):
        try:
            if self.browser is not None and self.browser.is_connected():
                self.browser.close()
        finally:
            self.manager.stop()
