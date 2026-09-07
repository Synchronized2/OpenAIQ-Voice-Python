from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from diagnostics import log


SAMPLE_RATE = 16_000
VAD_FRAME_SHIFT_SAMPLES = 160
VAD_FRAME_LENGTH_SAMPLES = 400
VAD_CHUNK_FRAMES = 10
CAPTURE_BLOCK_SAMPLES = VAD_CHUNK_FRAMES * VAD_FRAME_SHIFT_SAMPLES
VAD_CHUNK_SAMPLES = VAD_FRAME_LENGTH_SAMPLES + (VAD_CHUNK_FRAMES - 1) * VAD_FRAME_SHIFT_SAMPLES
_SENSEVOICE_TAG = re.compile(r"<\|[^|]+\|>")


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在: {path}")


def _resolve_vad_model_dir(path: Path) -> Path:
    candidate = path / "Stream-VAD"
    return candidate if candidate.is_dir() else path


def list_audio_devices() -> None:
    """Print input devices without loading the ASR and VAD models."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("缺少 sounddevice，请先运行 .\\setup.ps1") from exc

    output = str(sd.query_devices())
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(output.encode(encoding, errors="replace").decode(encoding))


class StreamingASR:
    """FireRed streaming endpoint detection followed by Paraformer recognition."""

    def __init__(
        self,
        model_dir: str | Path,
        vad_model_dir: str | Path,
        threads: int = 8,
        device: int | None = None,
        vad_threshold: float = 0.4,
        vad_end_silence: float = 0.7,
        pre_roll: float = 0.4,
        post_roll: float = 0.26,
        hotwords: str = "",
        debug_audio: bool = False,
    ) -> None:
        try:
            import sounddevice as sd
            import torch
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ASR 依赖，请在项目虚拟环境执行: "
                ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        project_dir = Path(__file__).resolve().parent
        vendor_dir = project_dir / "vendor" / "FireRedASR2S" / "fireredasr2s"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        try:
            from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        except ImportError as exc:
            raise RuntimeError(f"FireRedVAD 运行代码不可用: {vendor_dir}") from exc

        model_path = Path(model_dir).expanduser().resolve()
        vad_path = _resolve_vad_model_dir(Path(vad_model_dir).expanduser().resolve())
        _require_file(model_path / "model.pt", "Paraformer 模型")
        _require_file(model_path / "config.yaml", "Paraformer 配置")
        _require_file(vad_path / "model.pth.tar", "FireRedVAD 模型")
        _require_file(vad_path / "cmvn.ark", "FireRedVAD CMVN")

        self._sd = sd
        self.device = device
        self.vad_end_silence = max(0.2, float(vad_end_silence))
        self.pre_roll_samples = max(CAPTURE_BLOCK_SAMPLES, int(float(pre_roll) * SAMPLE_RATE))
        self.post_roll_samples = max(0, int(float(post_roll) * SAMPLE_RATE))
        self.hotwords = hotwords.strip()
        self.debug_audio = debug_audio
        thread_count = max(1, int(threads))
        torch.set_num_threads(thread_count)
        os.environ["OMP_NUM_THREADS"] = str(thread_count)

        print(f"ASR 模型: Paraformer-large ({model_path})")
        print(f"VAD 模型: FireRedVAD Stream-VAD ({vad_path})")
        self._vad = FireRedStreamVad.from_pretrained(
            str(vad_path),
            FireRedStreamVadConfig(
                use_gpu=False,
                speech_threshold=float(vad_threshold),
                pad_start_frame=5,
                min_speech_frame=8,
                max_speech_frame=2000,
                min_silence_frame=max(1, round(self.vad_end_silence * 100)),
            ),
        )
        self._recognizer = AutoModel(
            model=str(model_path),
            device="cpu",
            disable_update=True,
            disable_pbar=True,
        )

    def list_devices(self) -> None:
        output = str(self._sd.query_devices())
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(output.encode(encoding, errors="replace").decode(encoding))

    def transcribe(self, samples: np.ndarray) -> str:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        options: dict[str, object] = {
            "input": audio,
            "batch_size_s": 30,
            "use_itn": True,
        }
        if self.hotwords:
            options["hotword"] = self.hotwords
        result = self._recognizer.generate(**options)
        if not result:
            return ""
        text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
        return _SENSEVOICE_TAG.sub("", text).strip()

    def listen_once(
        self,
        stop_event: threading.Event | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_level: Callable[[float], None] | None = None,
    ) -> str:
        """Capture one utterance and stop at the first FireRedVAD endpoint."""
        if stop_event and stop_event.is_set():
            return ""
        chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

        def callback(indata, frames, callback_time, status) -> None:  # noqa: ANN001
            if status:
                print(f"\n音频状态: {status}", file=sys.stderr)
            try:
                chunks.put_nowait(indata[:, 0].copy())
            except queue.Full:
                print("\n警告: 音频处理跟不上采集速度，丢弃一帧", file=sys.stderr)

        pre_roll_blocks: deque[np.ndarray] = deque()
        pre_roll_size = 0
        utterance_blocks: list[np.ndarray] = []
        analysis_buffer = np.empty(0, dtype=np.float32)
        speech_active = False
        speech_started_at: float | None = None
        endpoint_reached = False
        self._vad.reset()

        print("\n请说话（连续静音后发送），Ctrl+C 退出。")

        with self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CAPTURE_BLOCK_SAMPLES,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        ):
            log.info("asr.capture.start device=%s", self.device)
            if on_status:
                on_status("正在聆听")
            while not endpoint_reached:
                if stop_event and stop_event.is_set():
                    if on_status:
                        on_status("已停止聆听")
                    self._vad.reset()
                    return ""
                try:
                    samples = chunks.get(timeout=0.2)
                except queue.Empty:
                    if on_level:
                        on_level(0.0)
                    continue

                if on_level:
                    rms = float(np.sqrt(np.mean(np.square(samples))))
                    on_level(min(1.0, rms * 12.0))

                if speech_active:
                    utterance_blocks.append(samples)
                else:
                    pre_roll_blocks.append(samples)
                    pre_roll_size += len(samples)
                    while pre_roll_size > self.pre_roll_samples + CAPTURE_BLOCK_SAMPLES:
                        pre_roll_size -= len(pre_roll_blocks.popleft())

                analysis_buffer = np.concatenate((analysis_buffer, samples))
                while len(analysis_buffer) >= VAD_CHUNK_SAMPLES:
                    vad_audio = np.clip(
                        analysis_buffer[:VAD_CHUNK_SAMPLES] * 32768,
                        -32768,
                        32767,
                    ).astype(np.int16)
                    frame_results = self._vad.detect_chunk(vad_audio)
                    analysis_buffer = analysis_buffer[CAPTURE_BLOCK_SAMPLES:]
                    for frame_result in frame_results:
                        if frame_result.is_speech_start and not speech_active:
                            log.info("asr.vad.speech_start")
                            speech_active = True
                            speech_started_at = time.perf_counter()
                            utterance_blocks = list(pre_roll_blocks)
                            if on_speech_start:
                                on_speech_start()
                            if self.debug_audio:
                                print(f"[VAD] 检测到语音，概率 {frame_result.smoothed_prob:.3f}")
                            if on_status:
                                on_status("检测到语音")
                        if frame_result.is_speech_end and speech_active:
                            log.info("asr.vad.endpoint")
                            endpoint_reached = True
                            if self.debug_audio:
                                print(f"[VAD] 端点确认，连续静音 {self.vad_end_silence:.2f}s")
                            break
                    if endpoint_reached:
                        break

        self._vad.reset()
        if on_level:
            on_level(0.0)
        if stop_event and stop_event.is_set():
            return ""
        if not utterance_blocks:
            return ""
        audio = np.concatenate(utterance_blocks)
        trailing_to_remove = max(
            0,
            int(self.vad_end_silence * SAMPLE_RATE) - self.post_roll_samples,
        )
        if trailing_to_remove and len(audio) > trailing_to_remove:
            audio = audio[:-trailing_to_remove]
        if on_status:
            on_status("正在识别")
        if on_partial:
            on_partial("正在识别…")
        started = time.perf_counter()
        text = self.transcribe(audio)
        log.info("asr.transcribe.end seconds=%.3f chars=%s", time.perf_counter() - started, len(text))
        if stop_event and stop_event.is_set():
            return ""
        elapsed = time.perf_counter() - speech_started_at if speech_started_at else 0.0
        if text:
            print(f"最终: {text}  [本句耗时 {elapsed:.2f}s]")
            if on_status:
                on_status("识别完成")
        elif self.debug_audio:
            print("[ASR] FireRedVAD 已检测到语音，但 Paraformer 没有返回文字。")
        return text
