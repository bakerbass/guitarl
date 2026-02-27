"""
Continuously-recording rolling audio buffer for RL step timing.

Records audio from the hardware device in a background thread into a
fixed-size ring buffer.  Provides onset detection and silence gating so
that each RL step can:

  1. Wait for the previous harmonic to fully decay (silence gate).
  2. Timestamp the robot action at the moment of OSC send.
  3. Find the actual note onset in the buffer relative to that timestamp.
  4. Extract an onset-aligned audio window for the CNN classifier.

This eliminates the cumulative desync caused by opening a new InputStream
on every step and removes dependence on the fixed CAPTURE_PRE_DELAY.
"""

import threading
import time
import logging
import numpy as np
import sounddevice as sd
from typing import Optional

logger = logging.getLogger(__name__)


class RollingAudioBuffer:
    """
    Thread-safe ring buffer that records audio continuously in the background.

    Timing model
    ------------
    The background thread tracks ``abs_write_pos``: the total number of samples
    written since ``start()`` was called.  Combined with ``t_stream_start``
    (monotonic wall-clock at stream open), any wall-clock time ``t`` maps to:

        elapsed_samples = int((t - t_stream_start) * device_sr)
        buffer_index    = elapsed_samples % buffer_samples

    because absolute sample N lives at ``buffer[N % buffer_samples]``.

    Parameters
    ----------
    device_id : int or None
        Sounddevice device index.  If None, silence is substituted and
        onset/silence methods return immediately (testing / offline mode).
    device_sr : int
        Hardware sample rate (e.g. 44100 or 48000).
    buffer_duration : float
        Ring buffer length in seconds (default 60.0).
    chunk_size : int
        Samples per sounddevice read call (default 2048).
    """

    def __init__(
        self,
        device_id: Optional[int],
        device_sr: int,
        buffer_duration: float = 60.0,
        chunk_size: int = 2048,
    ):
        self.device_id = device_id
        self.device_sr = device_sr
        self.chunk_size = chunk_size

        self.buffer_samples = int(buffer_duration * device_sr)
        self._buf = np.zeros(self.buffer_samples, dtype=np.float32)

        # Absolute write position (monotonically increasing, never wraps)
        self._abs_write_pos: int = 0
        # Monotonic time at which the stream was started
        self._t_stream_start: float = 0.0

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Set once the InputStream has been successfully opened for the first
        # time.  Callers can block on this before using the buffer.
        self._ready_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, open_timeout: float = 15.0) -> None:
        """Start the background recording thread and wait for it to open.

        Parameters
        ----------
        open_timeout : float
            Seconds to wait for the hardware stream to successfully open before
            raising ``RuntimeError``.  Set to 0 to return immediately (no wait).
            Default 15 s is generous enough for USB devices that take a moment
            to reinitialise after being released by another process.
        """
        if self._running:
            return
        self._running = True
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop, daemon=True, name="RollingAudioBuffer"
        )
        self._thread.start()
        logger.info(
            f"RollingAudioBuffer started (device={self.device_id}, "
            f"sr={self.device_sr}, buf={self.buffer_samples/self.device_sr:.0f}s)"
        )
        if self.device_id is None or open_timeout <= 0:
            return
        if not self._ready_event.wait(timeout=open_timeout):
            self._running = False
            raise RuntimeError(
                f"RollingAudioBuffer: audio device {self.device_id} did not open "
                f"within {open_timeout:.0f}s.  Is the Scarlett held by another "
                f"process?  Run:\n"
                f"  sudo fuser -k /dev/snd/*  (or  ./scripts/fix_audio.sh)"
            )

    def stop(self) -> None:
        """Signal the recording thread to stop and wait for it to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal recording loop
    # ------------------------------------------------------------------

    def _record_loop(self) -> None:
        if self.device_id is None:
            logger.warning("RollingAudioBuffer: no device, recording silence")
            self._t_stream_start = time.monotonic()
            while self._running:
                time.sleep(0.05)
            return

        MAX_RETRIES   = 10
        RETRY_DELAYS  = [1, 2, 4, 8, 8, 8, 8, 8, 8, 8]  # seconds between attempts
        attempt       = 0

        while self._running and attempt <= MAX_RETRIES:
            if attempt > 0:
                delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    f"RollingAudioBuffer: retrying device open "
                    f"(attempt {attempt}/{MAX_RETRIES}, waiting {delay}s)..."
                )
                time.sleep(delay)

            try:
                with sd.InputStream(
                    samplerate=self.device_sr,
                    channels=1,
                    device=self.device_id,
                    dtype="float32",
                    blocksize=self.chunk_size,
                ) as stream:
                    if attempt > 0:
                        logger.info(
                            f"RollingAudioBuffer: device reopened successfully "
                            f"(attempt {attempt})."
                        )
                    attempt = 0  # reset on successful open
                    self._ready_event.set()  # unblock start() / wait_until_ready()
                    self._t_stream_start = time.monotonic()
                    while self._running:
                        chunk, _ = stream.read(self.chunk_size)
                        samples = chunk[:, 0]
                        n = len(samples)
                        with self._lock:
                            pos = self._abs_write_pos % self.buffer_samples
                            end = pos + n
                            if end <= self.buffer_samples:
                                self._buf[pos:end] = samples
                            else:
                                # Wrap around
                                first = self.buffer_samples - pos
                                self._buf[pos:] = samples[:first]
                                self._buf[: n - first] = samples[first:]
                            self._abs_write_pos += n
            except Exception as exc:
                logger.error(f"RollingAudioBuffer recording error: {exc}")
                attempt += 1

        if attempt > MAX_RETRIES:
            logger.error(
                f"RollingAudioBuffer: gave up after {MAX_RETRIES} retries — "
                f"audio capture disabled for this run."
            )
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self):
        """Return a consistent (abs_write_pos, t_stream_start) snapshot."""
        with self._lock:
            return self._abs_write_pos, self._t_stream_start

    def _t_to_abs_sample(self, t: float, abs_write_pos: int, t_stream_start: float) -> int:
        """Convert wall-clock t to an absolute sample index (may be negative if before stream start)."""
        return int((t - t_stream_start) * self.device_sr)

    def get_audio_range(self, t_start: float, t_end: float) -> np.ndarray:
        """
        Extract audio between two monotonic wall-clock timestamps.

        If the requested range is partially or fully outside the buffer,
        the missing portion is zero-padded.

        Parameters
        ----------
        t_start, t_end : float
            Monotonic wall-clock times (``time.monotonic()``).

        Returns
        -------
        np.ndarray  shape=(n_samples,), dtype=float32
        """
        abs_write, t0 = self._snapshot()
        abs_start = self._t_to_abs_sample(t_start, abs_write, t0)
        abs_end   = self._t_to_abs_sample(t_end,   abs_write, t0)
        n_wanted  = max(abs_end - abs_start, 0)

        if n_wanted == 0:
            return np.zeros(0, dtype=np.float32)

        out = np.zeros(n_wanted, dtype=np.float32)

        # How many samples are available in the buffer?
        buf_oldest = abs_write - self.buffer_samples  # oldest sample index kept

        for i in range(n_wanted):
            abs_idx = abs_start + i
            if abs_idx < buf_oldest or abs_idx >= abs_write:
                # Outside buffer range — zero pad
                continue
            ring_idx = abs_idx % self.buffer_samples
            out[i] = self._buf[ring_idx]

        return out

    def _get_recent_audio(self, duration: float) -> np.ndarray:
        """Return the most recently recorded `duration` seconds."""
        abs_write, t0 = self._snapshot()
        n = int(duration * self.device_sr)
        out = np.zeros(n, dtype=np.float32)
        buf_oldest = abs_write - self.buffer_samples
        for i in range(n):
            abs_idx = abs_write - n + i
            if abs_idx < buf_oldest or abs_idx < 0:
                continue
            out[i] = self._buf[abs_idx % self.buffer_samples]
        return out

    # ------------------------------------------------------------------
    # Silence threshold calibration
    # ------------------------------------------------------------------

    def calibrate_silence_threshold(
        self,
        duration: float = 2.0,
        multiplier: float = 1.5,
    ) -> float:
        """
        Measure the ambient noise floor and return a suitable silence threshold.

        Records ``duration`` seconds of quiescent audio (no robot action should
        be sent during calibration), computes the median per-chunk RMS across
        20 ms windows, and returns ``median_rms * multiplier``.

        Using the median (rather than mean) makes the estimate robust against
        brief transients (e.g. room noise spikes) during the calibration window.

        Parameters
        ----------
        duration : float
            Seconds of ambient audio to sample (default 2.0).
        multiplier : float
            Scale factor above the noise floor that counts as "silent".
            A value of 3.0 means: signal must be < 3× the ambient RMS.

        Returns
        -------
        float
            Silence RMS threshold.  Always ≥ 1e-6 to prevent a zero threshold
            on extremely quiet or absent audio.
        """
        if self.device_id is None:
            logger.warning("calibrate_silence_threshold: no device, returning default 0.005")
            return 0.005

        # Block until the stream is actually open.  start() may already have
        # waited, but calibrate_silence_threshold can also be called later when
        # the stream drops and reconnects, so we wait here too.
        if not self._ready_event.wait(timeout=15.0):
            raise RuntimeError(
                "calibrate_silence_threshold: audio device never opened. "
                "Cannot calibrate silence threshold."
            )

        logger.info(
            f"Calibrating silence threshold: recording {duration:.1f}s of ambient audio..."
        )

        # Wait until we have at least `duration` seconds worth of data in the buffer.
        needed_samples = int(duration * self.device_sr)
        deadline = time.monotonic() + duration + 3.0  # generous slack
        while time.monotonic() < deadline:
            abs_write, _ = self._snapshot()
            if abs_write >= needed_samples:
                break
            time.sleep(0.05)
        else:
            logger.warning("calibrate_silence_threshold: timed out waiting for buffer data")

        audio = self._get_recent_audio(duration)

        chunk_size = int(0.02 * self.device_sr)  # 20 ms chunks
        rms_values = []
        for start in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[start : start + chunk_size]
            rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))

        if not rms_values:
            logger.warning("calibrate_silence_threshold: no chunks, returning default 0.005")
            return 0.005

        noise_floor = float(np.median(rms_values))
        threshold = max(noise_floor * multiplier, 1e-6)
        logger.info(
            f"  Noise floor (median RMS): {noise_floor:.6f}  "
            f"Threshold (×{multiplier}):  {threshold:.6f}"
        )
        return threshold

    # ------------------------------------------------------------------
    # Silence gate
    # ------------------------------------------------------------------

    def wait_for_silence(
        self,
        rms_threshold: float = 0.005,
        hold_duration: float = 0.5,
        timeout: float = 8.0,
        check_interval: float = 0.05,
    ) -> bool:
        """
        Block until the audio signal has been below ``rms_threshold`` for
        ``hold_duration`` continuous seconds.

        Parameters
        ----------
        rms_threshold : float
            Peak RMS per 20 ms sub-chunk that counts as silence.
        hold_duration : float
            Seconds that must all be silent before returning True.
        timeout : float
            Maximum wait in seconds; returns False if exceeded.
        check_interval : float
            Polling interval.

        Returns
        -------
        bool  True = silence confirmed, False = timed out.
        """
        if self.device_id is None:
            return True  # No device — always "silent"

        chunk_size = int(0.02 * self.device_sr)  # 20 ms chunks
        hold_samples = int(hold_duration * self.device_sr)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            recent = self._get_recent_audio(hold_duration + check_interval)
            if len(recent) < hold_samples:
                time.sleep(check_interval)
                continue

            # Check all 20 ms sub-chunks in the hold window
            window = recent[-hold_samples:]
            all_silent = True
            for start in range(0, len(window), chunk_size):
                sub = window[start : start + chunk_size]
                if len(sub) == 0:
                    continue
                rms = float(np.sqrt(np.mean(sub ** 2)))
                if rms > rms_threshold:
                    all_silent = False
                    break

            if all_silent:
                return True

            time.sleep(check_interval)

        logger.warning(
            f"wait_for_silence: timed out after {timeout:.1f}s "
            f"(threshold={rms_threshold:.4f}, hold={hold_duration:.2f}s)"
        )
        return False

    # ------------------------------------------------------------------
    # Onset detection
    # ------------------------------------------------------------------

    def wait_for_onset(
        self,
        after_time: float,
        search_window: float = 3.0,
        timeout: float = 5.0,
        threshold_factor: float = 8.0,
        bg_window: float = 0.1,
        chunk_dur: float = 0.02,
        fallback_delay: float = 0.5,
    ) -> float:
        """
        Wait for a note onset after ``after_time`` and return its wall-clock
        timestamp.

        Algorithm
        ---------
        1. Once ``bg_window`` seconds of audio past ``after_time`` are
           available, estimate background RMS from that baseline window.
        2. Step forward in 20 ms chunks; first chunk where
           ``rms > bg_rms * threshold_factor`` is the onset.
        3. If no onset found within ``timeout``, return
           ``after_time + fallback_delay`` with a warning.

        Parameters
        ----------
        after_time : float
            Monotonic timestamp of the robot action (OSC send).
        search_window : float
            Seconds after after_time to search for onset.
        timeout : float
            Maximum real-world wait before falling back.
        threshold_factor : float
            How many times above background RMS triggers onset.
        bg_window : float
            Seconds after after_time used to estimate background RMS.
        chunk_dur : float
            Duration of each energy chunk (seconds).
        fallback_delay : float
            Fallback offset from after_time if no onset found.

        Returns
        -------
        float  Monotonic wall-clock timestamp of detected onset.
        """
        if self.device_id is None:
            return after_time + fallback_delay

        chunk_samples = int(chunk_dur * self.device_sr)
        bg_samples    = int(bg_window * self.device_sr)
        deadline      = after_time + timeout
        bg_rms        = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            # How much audio past after_time do we have?
            available = now - after_time
            if available < bg_window + chunk_dur:
                time.sleep(0.01)
                continue

            # Get the full search window written so far
            t_end = min(after_time + search_window, now - 0.01)
            audio = self.get_audio_range(after_time, t_end)

            if len(audio) < bg_samples + chunk_samples:
                time.sleep(0.01)
                continue

            # Estimate background RMS from first bg_window seconds
            bg_chunk = audio[:bg_samples]
            if bg_rms is None:
                bg_rms = float(np.sqrt(np.mean(bg_chunk ** 2))) + 1e-8

            # Step through chunks after the background window
            for offset in range(bg_samples, len(audio) - chunk_samples, chunk_samples):
                sub = audio[offset : offset + chunk_samples]
                rms = float(np.sqrt(np.mean(sub ** 2)))
                if rms > bg_rms * threshold_factor:
                    onset_offset_s = offset / self.device_sr
                    onset_time = after_time + onset_offset_s
                    logger.debug(
                        f"Onset detected at +{onset_offset_s:.3f}s after action "
                        f"(rms={rms:.5f}, bg={bg_rms:.5f}, factor={rms/bg_rms:.1f}x)"
                    )
                    return onset_time

            time.sleep(0.02)

        fallback = after_time + fallback_delay
        logger.warning(
            f"wait_for_onset: no onset found within {timeout:.1f}s, "
            f"falling back to after_time + {fallback_delay:.2f}s"
        )
        return fallback
