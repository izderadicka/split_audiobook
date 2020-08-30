import unittest
import unittest.mock as mock
import io
import os
import split_audiobook
from split_audiobook import *  # pylint: disable=W0614

DUMMY_CHAPTERS = """Chapter 1,0,360
"Beers,tears and queers",360,1000
"One/Two:Three",1000,1:01:01.1
"""


class Tests(unittest.TestCase):

    def test_secs(self):
        s1 = secs_from_time("11.5")
        self.assertEqual(11.5, s1)
        s2 = secs_from_time("21:11.5")
        self.assertEqual(21*60+11.5, s2)
        s3 = secs_from_time("2:21:11.5")
        self.assertEqual(2*3600+21*60+11.5, s3)

    def test_chapters_file(self):
        cs = file_to_chapters_iter(io.StringIO(DUMMY_CHAPTERS), False)
        cs = list(cs)
        self.assertEqual(3, len(cs))
        self.assertEqual("Beers,tears and queers", cs[1][0])
        self.assertEqual("One-Two-Three", cs[2][0])
        self.assertEqual(360, cs[0][2])
        self.assertTrue(type(cs[0][1]) == float)
        self.assertEqual(3661.1, cs[2][2])

    def test_chapters_file2(self):
        with open("test_data/external_chapters.csv") as f:
            cs = file_to_chapters_iter(f, False)
            cs = list(cs)
        self.assertEqual(17, len(cs))
        self.assertEqual(cs[1][2], 29*60+46.1)
        self.assertTrue(cs[16][2] is None)

    def test_cue_file(self):
        with open("test_data/external_chapters.cue") as f:
            cs = file_to_chapters_iter(f, True)
            cs = list(cs)
        self.assertEqual(17, len(cs))
        self.assertEqual(cs[1][2], 29*60+46+8/75)
        self.assertTrue(cs[16][2] is None)

    def test_chapters_meta(self):
        with open("test_data/chapters.txt", "rb") as f:
            data = f.read()
        with mock.patch.object(split_audiobook, "_run_ffprobe_for_chapters") as mocked_run_ffprobe:
            mocked_run_ffprobe.return_value = data
            cs = meta_to_chapters_iter("some.mp3")
        cs = list(cs)
        self.assertEqual(59, len(cs))
        for l in cs:
            self.assertTrue(l[1] < l[2])

    def test_split_by_silence(self):
        with open('test_data/silences.txt', 'rb') as f:
            data = f.read()
        with mock.patch.object(split_audiobook.SilenceDetector, "_run_ffmpeg") as mocked_run_ffmpeg:
            mocked_run_ffmpeg.return_value = data
            opts = mock.MagicMock(length=1800, silence=30, silence_duration=2)
            chapters = calc_split_points("test_file.mp3", opts)
            chapters = list(chapters)
        self.assertEqual(16, len(chapters))
        prev_end = 0
        for _chap, start, end in chapters:
            self.assertEqual(prev_end, start)
            if end is not None:
                dur = end - start
                diff = abs(dur - opts.length)
                self.assertTrue(
                    diff < opts.length / 2, "diff %0.2f is smaller then half of length %0.2f" % (diff, opts.length / 2))
            prev_end = end

    def test_detect_silence(self):
        with open('test_data/silences.txt', 'rb') as f:
            data = f.read()
        with mock.patch.object(SilenceDetector, "_run_ffmpeg") as mocked_run_ffmpeg:
            mocked_run_ffmpeg.return_value = data
            d = SilenceDetector("somefile.mp3")
        s = list(d)
        self.assertTrue(len(s) > 50)

        s2 = d.find_after(1800)
        self.assertEqual(1813.24, s2[0][0])
        self.assertEqual(1815.3, s2[0][1])

        s1 = d.find_before(1800)
        self.assertEqual(1507.13, s1[0][0])
        self.assertEqual(1509.17, s1[0][1])


if __name__ == "__main__":
    unittest.main()
