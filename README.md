split_audiobook
===============

Simple python script (Python 3.5+), that wraps ffmpeg and ffprobe and enables to split large audiobook file into smaller parts.
There are basically 3 methods supported:

* if file contains chapters metadata, then it's split to chapters - it's probably the best case
* you can supply CSV file with chapters definition (name, start, end). You can create this file yourself, by listening to the original file (good  tool is audacity - detect silence (Analyze/Silence Finder) spots and listen around silences to find chapters beginnigs, write table in a spreadsheet - name, start, end and save as CSV)
* split into pieces of approximately same size (defined by `--length`) on silence moments in audio (for best listening expeperience). This method is used if none of previous two is available.

You can use `--dry` argument to see what actual split will be.

Resulting files are stored in directory, with same name as file (just without extension). Also cover image is extracted to that directory, if available in metadata.
By default script transcodes parts with Opus codec, which is my favorite audio codec for audio books. Result then can be easily used in my audiobooks server [audioserve](https://github.com/izderadicka/audioserve).
If you want to just split, but not to transcode use `--split-only` argument.

For full usage use `--help` argument.

And I guess you noticed you need to have [ffmpeg and ffprobe](https://www.ffmpeg.org/download.html) installed. And yeah, I tested only on Linux (Win/Mac users your are on your own, although I tried to make it as generic as possible).

Install
-------

It's just one script - so you can just [download](https://raw.githubusercontent.com/izderadicka/split_audiobook/master/split_audiobook.py) and run in python3 interpreter. Script is intentionaly using only python core modules.
Or you can install locally with `sudo pip3 install git+https://github.com/izderadicka/split_audiobook`.

License
-------

[MIT](https://opensource.org/licenses/MIT)