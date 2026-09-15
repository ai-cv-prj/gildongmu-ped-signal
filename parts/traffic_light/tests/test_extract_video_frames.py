import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "extract_video_frames", Path(__file__).parents[1] / "extract_video_frames.py"
)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


class ExtractVideoFramesTests(unittest.TestCase):
    def test_defaults(self):
        args = extractor.parse_args(["--source", "data"])
        self.assertEqual(args.sample_fps, 3.0)
        self.assertEqual(args.jpeg_quality, 95)
        self.assertIsNone(args.max_frames_per_video)

    def test_numeric_validation(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(extractor.argparse.ArgumentTypeError):
                extractor.positive_float(value)
        for value in ("0", "101"):
            with self.assertRaises(extractor.argparse.ArgumentTypeError):
                extractor.jpeg_quality(value)

    def test_find_videos_is_recursive_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "b.MOV").touch()
            (root / "nested/a.mp4").touch()
            (root / "ignore.txt").touch()
            videos = extractor.find_videos(root)
            self.assertEqual([path.name for path in videos], ["b.MOV", "a.mp4"])

    def test_output_must_be_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                extractor.ensure_empty_output(output)

    def test_duplicate_stems_receive_unique_names(self):
        paths = [Path("one/video.mp4"), Path("two/video.mov")]
        self.assertEqual(extractor.unique_output_names(paths), ["video", "video_2"])


if __name__ == "__main__":
    unittest.main()
