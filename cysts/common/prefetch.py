"""Download studies ahead of the GPU, so fetching never blocks analysis.

THE SHAPE OF THE PROBLEM. Per study the pipeline uses three different resources:

    network  ~15 s   download
    CPU      ~37 s   dcm2niix, crop write, measurement
    GPU      ~83 s   organ segmentation + cyst model

Run serially that is ~135 s. The three never contend, so overlapping them makes
throughput the SLOWEST stage rather than their sum -- ~83 s, set by the GPU. Downloading
study N+1 while the GPU is busy with study N is the cheapest part of that, and costs
nothing but disk.

WHY A BOUNDED QUEUE. An unbounded prefetcher fills the disk: these studies average
~190 MB and a 752-study run would stage 140 GB ahead of a GPU that consumes one every
83 s. `depth` caps how many completed-but-unconsumed studies may sit on disk.

WHY NOT MORE DOWNLOAD WORKERS. Measured during the 752-study bulk download: the archive
throttles per connection at ~150 KB/s and starts returning 429 above roughly 40
connections. Eight concurrent streams gave 13.7 MB/s aggregate while a single stream gave
19.7 MB/s -- more workers made it SLOWER. Two or three is the useful range; the default
is 2 and raising it is unlikely to help.

A FAILED PREFETCH IS NOT A FAILED STUDY. The error is carried on the queue and raised in
the consumer, so a cold-archive 503 surfaces against the right study rather than killing
the worker.
"""
import os
import queue
import threading

from . import download_dicoms, paths


class Prefetcher:
    """Iterate study_iuids, yielding (iuid, dicom_dir) once each is on disk."""

    def __init__(self, study_iuids, env="prod", depth=2, workers=2):
        self.ids = list(study_iuids)
        self.env = env
        self.depth = max(1, depth)
        self.workers = max(1, min(workers, 3))      # see module docstring
        self._out = queue.Queue(maxsize=self.depth)
        self._todo = queue.Queue()
        for i, s in enumerate(self.ids):
            self._todo.put((i, s))
        self._results = {}
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._threads = []

    def _work(self):
        while not self._stop.is_set():
            try:
                idx, iuid = self._todo.get_nowait()
            except queue.Empty:
                return
            dest = os.path.join(paths.DICOMS, iuid)
            try:
                info = download_dicoms.fetch(iuid, dest, env=self.env)
                res = (iuid, dest, info, None)
            except Exception as e:
                res = (iuid, dest, None, e)
            with self._lock:
                self._results[idx] = res
                self._lock.notify_all()

    def __iter__(self):
        for t in range(self.workers):
            th = threading.Thread(target=self._work, daemon=True, name=f"prefetch-{t}")
            th.start()
            self._threads.append(th)
        try:
            for idx in range(len(self.ids)):
                with self._lock:
                    # bound how far ahead the fetchers may run: completed-but-unconsumed
                    while (len(self._results) - idx) > self.depth and idx not in self._results:
                        self._lock.wait(0.5)
                    while idx not in self._results:
                        self._lock.wait(0.5)
                    iuid, dest, info, err = self._results.pop(idx)
                yield iuid, dest, info, err
        finally:
            self._stop.set()

    def close(self):
        self._stop.set()
