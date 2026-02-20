"""
Asynchronous recorder for successful harmonic actions during RL training.

Each successful harmonic (harmonic_prob > success_threshold) produces:
  - A WAV file containing the captured audio at the device's native sample rate
  - A JSON sidecar with the full action, reward, and classification metadata

record() is non-blocking: it copies the audio array onto an internal queue
and returns immediately so RL step() timing is never affected.  A single
daemon background thread drains the queue and handles all disk I/O.

--- Researcher workflow ---

After a training run, the successes/ directory contains paired .wav/.json
files ready for review:

  successes/
    000001_20260219_143201_str2_fret7.12_torque68.wav
    000001_20260219_143201_str2_fret7.12_torque68.json
    ...

Each JSON has a 'suggested_label' field (default 'harmonic') and a
'reviewed' flag (default False).  The researcher listens to each clip,
sets 'suggested_label' to 'harmonic', 'dead_note', or 'general_note'
as appropriate, and flips 'reviewed' to True.

Reviewed clips can then be fed back into HarmonicsClassifier retraining
to improve accuracy with real robot data:

  python run_build_dataset.py --from-successes ./runs/RUN/successes/
"""

import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SuccessRecorder:
    """
    Thread-safe, non-blocking recorder for successful harmonic actions.

    Usage:
        recorder = SuccessRecorder(output_dir)
        recorder.record(audio_array, metadata_dict)   # instant, non-blocking
        recorder.close()                              # drain + stop at shutdown

    The metadata dict must include 'device_sr' (the WAV sample rate).
    The env sets this automatically from reward_calc.device_sr.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue()
        self._count = 0
        self._count_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._worker, name="SuccessRecorder", daemon=True
        )
        self._thread.start()
        logger.info(f"[SuccessRecorder] Saving to {self.output_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, audio: np.ndarray, metadata: dict) -> None:
        """
        Non-blocking.  Enqueue audio + metadata for background disk write.

        The audio array is copied immediately so the caller can safely
        reuse or discard it after this call returns.

        Args:
            audio:    1-D or 2-D float32 audio captured at self.sample_rate.
            metadata: Dict of action/reward info written as a JSON sidecar.
        """
        self._queue.put((audio.copy(), dict(metadata)))

    def close(self) -> None:
        """Drain all pending writes then stop the background thread."""
        self._queue.join()      # wait for every enqueued item to be processed
        self._queue.put(None)   # sentinel: stop worker
        self._thread.join(timeout=10)
        logger.info(f"[SuccessRecorder] Closed. {self._count} recordings saved.")

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            audio, metadata = item
            try:
                self._write(audio, metadata)
            except Exception as exc:
                logger.error(f"[SuccessRecorder] Write error: {exc}", exc_info=True)
            finally:
                self._queue.task_done()

    def _write(self, audio: np.ndarray, metadata: dict) -> None:
        import soundfile as sf

        sr = metadata.get("device_sr", 44100)

        with self._count_lock:
            self._count += 1
            idx = self._count

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fret = metadata.get("fret_position", 0.0)
        torque = metadata.get("torque", 0)
        string_idx = metadata.get("string_index", 0)
        stem = f"{idx:06d}_{ts}_str{string_idx}_fret{fret:.2f}_torque{torque:.0f}"

        wav_path = self.output_dir / f"{stem}.wav"
        json_path = self.output_dir / f"{stem}.json"

        # Write WAV — ensure mono float32
        sf.write(str(wav_path), audio.flatten().astype(np.float32), sr, subtype="FLOAT")

        # Write JSON sidecar — include everything needed to re-label the clip
        sidecar = dict(metadata)
        sidecar["wav_file"] = wav_path.name
        sidecar["sample_rate"] = sr
        # Researcher-editable fields:
        sidecar["suggested_label"] = "harmonic"   # override if clip sounds wrong
        sidecar["reviewed"] = False               # flip to True after listening
        json_path.write_text(json.dumps(sidecar, indent=2, default=str))

        logger.debug(f"[SuccessRecorder] Saved {stem}")
