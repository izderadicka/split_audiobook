#!/usr/bin/python3

import argparse
import subprocess
import logging
import sys
import os
import csv
import re
import concurrent.futures as futures
import multiprocessing
import shutil
from math import floor
from functools import reduce

log = logging.getLogger()

__version__ = "0.3.1"

ABOUT = """
Splits large audiobook files into smaller parts which are then optionally encoded with Opus codec.
Split points are either chapters defined in the audiobook, or supplied in external CSV file,
or split happens in silence periods in approximately same distance (--length).
Use --dry option to see how audio will be split, without actual conversion or --write-chapters 
to write chapters into separate file in simple CSV format.
Requires ffmpeg and ffprobe version v >= 2.8.11
Supports input formats m4a, m4b, mp3, aax (mka should also work but not tested)
"""


def opus_params(quality):
    def op(bitrate, cutoff, quality, mono):
        params = ["-acodec", "libopus"]
        if mono:
            params.extend(["-ac", "1"])
        params.extend(["-b:a",  "%dk" % bitrate,  "-vbr",  "on",
                       "-compression_level",  "%d" % quality,  "-application", "audio"])
        if cutoff:
            params.extend(["-cutoff", "%d" % (cutoff*1000)])

        return params
    return op(*OPUS_PARAMS[quality])


OPUS_PARAMS = {
    # bitrate in kbps, cutoff in kHz, quality (1-10), mono dowmix
    "top": [64, 20, 10, False],
    "high": [48, 12, 10, False],
    "normal":  [32, 12, 10, True],
    "low": [24, 8, 10, True]
}


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=ABOUT,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", metavar="FILE", nargs="+",
                        help="audiobook files")
    parser.add_argument("--search-dir", action="store_true",
                        help="if FILE argument is directory it will recusivelly search for audio files and split them")
    parser.add_argument("--debug", action="store_true",
                        help="debug logging")
    parser.add_argument("--delete", action="store_true",
                        help="delete original file after split and conversion")
    parser.add_argument("-s", "--silence", type=int, default=30,
                        help="silence level in -x dB from max level")
    parser.add_argument("--silence-duration", type=float, default=2,
                        help="minimal duration of silence, default works fine, so change only, if necessary")
    parser.add_argument("--dry", action="store_true",
                        help="dry run, just prints calculated parts and exits")
    parser.add_argument("--write-chapters", action="store_true",
                        help="instead of spliting file, it just writes chapters into original_file.chapters CSV file")
    parser.add_argument("--cue-format", action="store_true",
                        help="use cue format instead of CSV to read and write chapters")
    parser.add_argument("-o", "--split-only", action="store_true",
                        help="do not transcode, just split to parts using same audio codec")
    parser.add_argument("--ignore-chapters", action="store_true",
                        help="ignores chapters metadata, if they are pressent")
    parser.add_argument("--remove", action="store_true",
                        help="remove existing directory of splitted files")
    parser.add_argument("-q", "--quality", choices=["low", "normal", "high", "top"], default="high",
                        help="opus codec quality params")
    parser.add_argument("-l", "--length", type=float, default=1800,
                        help="duration of split segment in seconds (in case chapers are not available)")
    parser.add_argument("-c", "--chapters", type=argparse.FileType("r", encoding="utf-8", errors="replace"),
                        help="CSV file with chapters information, each line should contain: chapter_name,start_in_secs,end_in_secs  (optionaly start and end can be in form hh:mm:ss.m)")
    parser.add_argument("--activation-bytes",
                        help="activation bytes for aax format")
    parser.add_argument("--version", action="version", version=__version__,
                        help="shows version")
    return parser.parse_args(args)


def test_ff():
    return test_exe("ffmpeg") and test_exe("ffprobe")


def test_exe(name):
    import sys
    if sys.platform == "win32" and not name.endswith(".exe"):
        name += ".exe"
    for p in os.environ.get("PATH", "").split(os.pathsep):
        exe = os.path.join(p, name)
        if os.access(exe, os.X_OK):
            return True


def main(args=sys.argv[1:]):
    opts = parse_args(args)
    logging.basicConfig(level=logging.DEBUG if opts.debug else logging.INFO)
    log.debug("Started with arguments: %s", opts)

    if not test_ff():
        log.fatal("ffmpeg or ffprobe not installed")
        sys.exit(2)
    if opts.chapters and len(opts.files) > 1:
        log.fatal("extenal chapters file can be used only for one audiobook file")

    pool = futures.ThreadPoolExecutor(multiprocessing.cpu_count())

    for fname in opts.files:
        if os.path.isdir(fname):
            if opts.search_dir:
                files = []
                for dirpath, _dirnames, filenames in os.walk(fname):
                    for f in filenames:
                        ext = os.path.splitext(f)[1]
                        if ext in (".mp3", ".m4b", ".m4a", ".mka", ".aax"):
                            files.append(os.path.join(dirpath, f))
                for f in files:
                    try:
                        split_file(f, pool, opts)
                    except:
                        log.exception("Error during splitting file %s", f)
            else:
                sys.exit("Arguments must be file, but you can force directory with --search-dir")
        else:
            try:
                split_file(fname, pool, opts)
            except:
                log.exception("Error during splitting file %s", fname)
    log.debug("All files analyzed")
    pool.shutdown()
    log.debug("Done")


def print_chapters(chapters, opts):
    for i, (chap, start, end) in enumerate(chapters):
        if end is None:
            print("%03d - %s  (%0.2f - end)" % (i, chap, start))
        else:
            print("%03d - %s  (%0.2f - %0.2f dur: %0.2f)" %
                  (i, chap, start, end, end-start))


def write_chapters(chapters, audio_file, cue_format):
    if not cue_format:
        with open(audio_file+".chapters", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('title', 'start', 'end'))
            writer.writerows(chapters)
    else:
        with open(audio_file+".cue", 'w', encoding='utf-8') as f:
            ext = os.path.splitext(audio_file)[1][1:].upper()
            f.write("FILE \"{filename}\" {ext}\n".format(filename=os.path.basename(audio_file), ext=ext))
            for track, (chap, start, _) in enumerate(chapters, 1):
                f.write("TRACK {track} AUDIO\n".format(track=track))
                f.write("  TITLE \"{title}\"\n".format(title=chap))
                f.write("  INDEX 01 {start}\n".format(start=cue_time_from_secs(start)))


EXT_MAP = {".m4b": ".m4a", ".aax": ".m4a"}


def map_ext(ext):
    mapped = EXT_MAP.get(ext)
    return mapped if mapped else ext


def split_file(fname, pool, opts):
    log.debug("Processing file %s", fname)
    base_dir, base_name = os.path.split(fname)
    base_name, format_ext = os.path.splitext(base_name)
    dest_dir = os.path.join(base_dir, base_name)

    if format_ext == ".aax":
        if not opts.activation_bytes:
            activation_file = os.path.expanduser("~/.audible_activation_bytes")
            if os.access(activation_file, os.R_OK):
                opts.activation_bytes = open(activation_file).read().strip()
        if not opts.activation_bytes or len(opts.activation_bytes) != 8:
            raise Exception("For aax file activation bytes must be provided and activation bytes must be 8 chars long")
    elif opts.activation_bytes and format_ext != ".aax":
        opts.activation_bytes = None

    chapters = None
    if opts.chapters:
        chapters = file_to_chapters_iter(opts.chapters, opts.cue_format)
    elif not opts.ignore_chapters:
        chapters = meta_to_chapters_iter(fname)

    if not chapters:
        chapters = calc_split_points(fname, opts)

    if opts.dry:
        print("File %s chapters" % fname)
        print_chapters(chapters, opts)
        return

    if opts.write_chapters:
        write_chapters(chapters, fname, opts.cue_format)
        return

    if os.path.exists(dest_dir):
        if opts.remove:
            shutil.rmtree(dest_dir)
        else:
            log.warning("Directory %s exists skipping split", dest_dir)
            return

    os.mkdir(dest_dir)

    op = opus_params(opts.quality) if not opts.split_only else [
        "-acodec", "copy"]
    ext = ".opus" if not opts.split_only else map_ext(format_ext)
    chapters = list(chapters)
    digits = len(str(len(chapters)))
    for i, (chap, start, end) in enumerate(chapters):
        pool.submit(transcode_chapter, fname, dest_dir, op, ext,
                    digits, i, chap, start, end, opts.activation_bytes)
    pool.submit(extract_cover, fname, dest_dir)


def extract_cover(fname, dest_dir):
    log.debug("extracting cover from file %s", fname)
    params = ["ffmpeg", "-v", "error", "-nostdin"]
    params.extend(["-i", fname, ])
    out_file = os.path.join(dest_dir, "cover.jpg")
    params.append(out_file)
    p = subprocess.Popen(params, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    _out, err = p.communicate()
    if p.returncode != 0:
        log.debug("Error extracting cover from file %s, return code %d\nStderr:\n%s",
                  fname, p.returncode, err)


def transcode_chapter(fname, dest_dir, op, ext, digits, i, chap, start, end, activation_bytes=None):
    log.debug("transcoding chapter %s", repr((dest_dir, i, chap, start, end)))
    try:
        params = ["ffmpeg", "-v", "error", "-nostdin"]
        if activation_bytes:
            params.extend(["-activation_bytes", activation_bytes])
        params.extend(["-i", fname, "-vn", "-ss", "%0.2f" % start])
        if not end is None:
            params.extend(["-to", "%0.2f" % end])
        params.extend(op)
        params.extend(["-metadata",  'title="%s"' % chap,
                       "-metadata", 'track="%d"' % (i+1)
                       ])
        out_file = os.path.join(dest_dir, "{i:0{digits}} - {chap}{ext}".format(i=i, digits=digits, chap=chap, ext=ext))
        params.append(out_file)
    except Exception as e:
        log.exception("Params preparation exception")
        raise e
    try:
        p = subprocess.Popen(params, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        _out, err = p.communicate()
    except:
        log.exception(
            "Transcoding of chapter %d - %s of file %s failed",  i, chap, fname)
        return
    log.debug("Finished transcoding chapter %d -%s of file %s", i, chap, fname)
    if p.returncode != 0:
        log.error("Error transcoding chapter %d - %s of file %s return code: %d\nStderr:\n%s",
                  i, chap, fname, p.returncode, err)


ELASTICITY = 1/3


def calc_split_points(fname, opts):
    def split_point(start, end):
        end_point = end - 0.5
        if end_point < start:
            end_point = start + (end-start)/2
        return end_point
    silences = SilenceDetector(fname, opts.silence, opts.silence_duration)
    total_duration = silences.total_duration
    prev_split = 0
    index = 0
    while not prev_split is None and prev_split < total_duration:
        next_split = prev_split + opts.length
        if total_duration - next_split < ELASTICITY * opts.length:
            # dont split only small part will remain
            next_split = total_duration
        else:
            silence = silences.find_after(next_split)
            if silence:
                (start, end), silence_index = silence
                split_at = split_point(start, end)
                diff = split_at - next_split
                assert diff > -1
                if diff > ELASTICITY * opts.length:
                    log.debug(
                        "Cannot find good split point after %0.2f diff %0.2f, will look for one before", next_split, diff)
                    if silence_index > 0:
                        maybe_split = split_point(*silences[silence_index-1])
                        diff = next_split - maybe_split
                        if diff < ELASTICITY * opts.length:
                            log.debug(
                                "Find good split below before %0.2f diff %0.2f", next_split, -diff)
                            split_at = maybe_split

                next_split = split_at
            else:
                next_split = total_duration
        yield ("Part %d" % index, prev_split, next_split)
        prev_split = next_split
        index += 1


def safe_name(fname):
    fname = fname.strip()
    for c in ":/":
        fname = fname.replace(c, "-")
    return fname


def secs_from_time(t):
    data = t.split(":")
    data.reverse()
    data = map(float, data)
    secs = reduce(lambda x, y: (x[0]+x[1]*y, x[1]*60), data, (0, 1))
    return secs[0]


def cue_time_from_secs(t):
    m = int(t / 60)
    s = int(t - m * 60)
    f = floor(t % 1 * 75)
    return "{m}:{s}:{f:02}".format(m=m, s=s, f=f)


def file_to_chapters_iter(f, cue_format):
    if not cue_format:
        has_header = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)
        reader = csv.reader(f)

        if has_header:
            next(reader, None)

        def format_line(l):
            if len(l) < 3:
                raise Exception("Chapters file lines must have at least 3 fields")
            return safe_name(l[0]), secs_from_time(l[1]), secs_from_time(l[2]) if l[2] else None
        return map(format_line, reader)
    else:
        cue_lines = f.read().splitlines()
        chapters = []
        for line in cue_lines:
            if line.startswith("TRACK "):
                chapters.append({})
            if re.match(r"\s+TITLE ", line):
                chapters[-1]["title"] = safe_name(" ".join(line.strip().split(" ")[1:]).replace("\"", ""))
            if re.match(r"\s+INDEX 01 ", line):
                t = list(map(int, " ".join(line.strip().split(" ")[2:]).replace("\"", "").split(":")))
                chapters[-1]["start"] = 60 * t[0] + t[1] + t[2] / 75.0
                if len(chapters) > 1:
                    chapters[-2]["end"] = chapters[-1]["start"]
        chapters[-1]["end"] = None
        return map(lambda t: tuple(t.values()), chapters)


def _run_ffprobe_for_chapters(f):
    p = subprocess.Popen(["ffprobe",
                          "-v", "error",
                          "-print_format", "compact=nokey=1", "-show_chapters",
                          f
                          ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE
                         )
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception(
            "Error reading chapter meta, process return code: %d\nStdErr:\n%s", p.returncode, err)
    if len(out) == 0:
        return None
    return out


def meta_to_chapters_iter(f):
    out = _run_ffprobe_for_chapters(f)
    if not out:
        return
    lines = out.decode('utf-8').splitlines()

    def format_line(l):
        items = l.split("|")
        start, end, chap = items[4], items[6], items[7]
        return safe_name(chap), float(start), float(end)

    return map(format_line, lines)


class SilenceDetector:
    def __init__(self, fname, db=30, duration=2):
        self.fname = fname
        self.duration = duration
        self.total_duration = None
        db = round(db)
        self.db = db if db > 0 else -db
        self._start = None
        self._silences = []
        self._detect()

    def __len__(self):
        return self._silences.__len__()

    def __getitem__(self, idx):
        return self._silences[idx]

    def __iter__(self):
        return self._silences.__iter__()

    def find_before(self, dur, from_index=0):
        next = self.find_after(dur, from_index)
        if next and next[1] > 0:
            prev_index = next[1]-1
            return self._silences[prev_index], prev_index
        else:
            # there is no next so previous must be last in list
            if self._silences:
                prev_index = len(self._silences) - 1
                return self._silences[prev_index], prev_index

    def find_after(self, dur, from_index=0):
        for i, x in enumerate(self._silences[from_index:]):
            if x[0] >= dur or (x[0] < dur and x[1] >= dur):
                return x, i+from_index

    LINE_RE = re.compile(r"\[silencedetect .+\]")
    START_RE = re.compile(r"silence_start: (\d+\.?\d*)")
    END_RE = re.compile(r"silence_end: (\d+\.?\d*)")
    DURATION_RE = re.compile("Duration: ([0-9:.]+)")

    def _run_ffmpeg(self):
        p = subprocess.Popen(["ffmpeg",
                              "-v", "info",
                              "-i", self.fname,
                              "-af", "silencedetect=n=-%ddB:d=%0.2f" % (
                                  self.db, self.duration),
                              "-f", "null", "-"
                              ],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE
                             )
        _out, err = p.communicate()
        if p.returncode != 0:
            raise Exception(
                "Error detecting silence parts: %d\nStdErr:\n%s", p.returncode, err)
        return err

    def _detect(self):
        data = self._run_ffmpeg()
        data = data.decode("utf-8", "replace").splitlines().__iter__()
        for l in data:
            m = self.DURATION_RE.search(l)
            if m:
                self.total_duration = secs_from_time(m.group(1))
                break
        if self.total_duration is None:
            raise Exception("Cannot get total duration from media file")

        data = filter(lambda x: self.LINE_RE.match(x), data)
        for item in data:
            if self._start is None:
                m = self.START_RE.search(item)
                if m:
                    start = float(m.group(1))
                    self._start = start
            else:
                m = self.END_RE.search(item)
                if m:
                    end = float(m.group(1))
                    assert self._start <= end
                    self._silences.append((self._start, end))
                    self._start = None
        # last start is at end of file - use it to correct total_duration
        if self._start and self._start > self.total_duration:
            self.total_duration = self._start


if __name__ == "__main__":
    main()
