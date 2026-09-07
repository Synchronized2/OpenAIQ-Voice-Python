from __future__ import annotations

import argparse
import msvcrt
import sys
import threading
import time
from pathlib import Path

import config
from agent import LocalAgent, is_voice_exit_phrase, parse_local_intent
from asr import StreamingASR, list_audio_devices
from chat import ChatSession
from diagnostics import configure_logging
from tts import EdgeSpeaker


class NextQuestionListener:
    def __init__(self, on_interrupt) -> None:
        self.interrupted = threading.Event()
        self._finished = threading.Event()
        self._on_interrupt = on_interrupt
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self.question: str | None = None

    def start(self) -> None:
        self.interrupted.clear()
        self._finished.clear()
        self._thread = threading.Thread(target=self._watch, name="keyboard-interrupt", daemon=True)
        self._thread.start()

    def close(self) -> str:
        self._finished.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        with self._lock:
            return "" if self.question else "".join(self._buffer)

    def _watch(self) -> None:
        while not self._finished.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in {"\r", "\n"}:
                    with self._lock:
                        question = "".join(self._buffer).strip()
                    if not question:
                        continue
                    self.question = question
                    self.interrupted.set()
                    self._on_interrupt()
                    print(f"\n[收到新问题] {question}")
                    return
                if key == "\x1b":
                    self.interrupted.set()
                    self._on_interrupt()
                    return
                if key == "\b":
                    with self._lock:
                        if self._buffer:
                            self._buffer.pop()
                    continue
                if key in {"\x00", "\xe0"} and msvcrt.kbhit():
                    msvcrt.getwch()
                    continue
                if key.isprintable():
                    with self._lock:
                        self._buffer.append(key)
            time.sleep(0.03)


def choose_chat_model(requested: str | None = None) -> str:
    if requested:
        return requested
    print("\n选择聊天模型（直接回车使用默认项）：")
    for index, (model, description) in enumerate(config.CHAT_MODEL_OPTIONS, start=1):
        default_mark = " [默认]" if model == config.CHAT_MODEL else ""
        print(f"  {index}. {model} - {description}{default_mark}")
    while True:
        choice = input("模型编号: ").strip()
        if not choice:
            return config.CHAT_MODEL
        if choice.isdigit() and 1 <= int(choice) <= len(config.CHAT_MODEL_OPTIONS):
            return config.CHAT_MODEL_OPTIONS[int(choice) - 1][0]
        print("请输入列表中的编号。")


def run_text_mode(model: str) -> int:
    print("=" * 58)
    print(" OpenAIQ Voice Python")
    print(" Text -> GPT Chat -> Edge TTS")
    print("=" * 58)
    print("输入消息后按回车发送，输入 q 退出。")
    print("AI 生成或朗读期间可直接输入下一问题并回车；Esc 只打断。")

    try:
        chat = ChatSession(model=model)
        speaker = EdgeSpeaker()
        agent = LocalAgent()
    except Exception as exc:
        print(f"\n[启动失败] {exc}")
        return 1

    pending_question: str | None = None
    pending_prefix = ""
    try:
        while True:
            if pending_question is not None:
                user_text = pending_question
                pending_question = None
            elif pending_prefix:
                suffix = input(f"\n你：{pending_prefix}")
                user_text = f"{pending_prefix}{suffix}".strip()
                pending_prefix = ""
            else:
                user_text = input("\n你：").strip()
            if user_text.lower() in {"q", "quit", "exit"} or is_voice_exit_phrase(user_text):
                break
            if not user_text:
                continue

            action = parse_local_intent(user_text)
            if action is not None:
                result = agent.execute(action)
                print(f"\nAI：{result.message}")
                try:
                    speaker.speak(result.message)
                except Exception as exc:
                    print(f"[TTS 失败] {exc}")
                continue

            listener = NextQuestionListener(speaker.stop)
            try:
                speaker.begin_response()
                listener.start()
                _, generation_interrupted = chat.reply(
                    user_text,
                    on_sentence=speaker.enqueue,
                    should_stop=listener.interrupted.is_set,
                )
                if generation_interrupted:
                    if listener.question:
                        pending_question = listener.question
                    else:
                        print("\n[已打断当前回复]")
                    continue
                print("[TTS] 可直接输入下一问题并回车打断；Esc 只停止当前回复。")
                if not speaker.wait_interruptible(listener.interrupted):
                    if listener.question:
                        pending_question = listener.question
                    else:
                        print("\n[已打断当前回复]")
            except KeyboardInterrupt:
                speaker.stop()
                print("\n[已中断当前回复]")
            except Exception as exc:
                print(f"\n[请求失败] {exc}")
            finally:
                partial = listener.close()
                if pending_question is None and partial:
                    pending_prefix = partial
    except (KeyboardInterrupt, EOFError):
        print("\n正在退出…")
    finally:
        speaker.close()
    return 0


def run_voice_mode(
    model: str,
    model_dir: str | Path,
    vad_model_dir: str | Path,
    device: int | None,
    threads: int,
    vad_threshold: float,
    vad_end_silence: float,
    pre_roll: float,
    post_roll: float,
    hotwords: str,
    debug_audio: bool = False,
    list_devices: bool = False,
) -> int:
    print("=" * 58)
    print(" OpenAIQ Voice Python")
    print(" FireRedVAD -> Paraformer-large -> OpenAI Chat -> Edge TTS")
    print("=" * 58)
    print("说话后连续静音会自动发送；TTS 播放完成后继续监听。")

    try:
        if list_devices:
            list_audio_devices()
            return 0
        asr = StreamingASR(
            model_dir=model_dir,
            vad_model_dir=vad_model_dir,
            threads=threads,
            device=device,
            vad_threshold=vad_threshold,
            vad_end_silence=vad_end_silence,
            pre_roll=pre_roll,
            post_roll=post_roll,
            hotwords=hotwords,
            debug_audio=debug_audio,
        )
        chat = ChatSession(model=model)
        speaker = EdgeSpeaker()
        agent = LocalAgent()
    except Exception as exc:
        print(f"\n[启动失败] {exc}")
        return 1

    try:
        while True:
            user_text = asr.listen_once()
            if not user_text:
                continue
            if user_text.casefold() in {"q", "quit", "exit", "退出"} or is_voice_exit_phrase(user_text):
                break
            print(f"你：{user_text}")
            action = parse_local_intent(user_text)
            if action is not None:
                result = agent.execute(action)
                print(f"AI：{result.message}")
                try:
                    speaker.speak(result.message)
                except Exception as exc:
                    print(f"[TTS 失败] {exc}")
                continue
            try:
                speaker.begin_response()
                _, interrupted = chat.reply(user_text, on_sentence=speaker.enqueue)
                if not interrupted:
                    speaker.wait()
            except KeyboardInterrupt:
                speaker.stop()
                print("\n[已中断当前回复]")
            except Exception as exc:
                speaker.stop()
                print(f"\n[请求失败] {exc}")
    except (KeyboardInterrupt, EOFError):
        print("\n正在退出…")
    finally:
        speaker.close()
    return 0


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="OpenAIQ 语音助手")
    parser.add_argument("--text", action="store_true", help="使用原有键盘输入模式")
    parser.add_argument("--model", help="直接指定聊天模型，跳过启动选择菜单")
    parser.add_argument("--list-devices", action="store_true", help="列出麦克风设备")
    parser.add_argument("--model-dir", type=Path, default=Path(config.ASR_MODEL_DIR))
    parser.add_argument("--vad-model-dir", type=Path, default=Path(config.ASR_VAD_MODEL_DIR))
    parser.add_argument("--threads", type=int, default=config.ASR_THREADS)
    parser.add_argument("--device", type=int, default=config.ASR_DEVICE)
    parser.add_argument("--vad-threshold", type=float, default=config.ASR_VAD_THRESHOLD, help="FireRedVAD 语音概率阈值")
    parser.add_argument("--vad-end-silence", type=float, default=config.ASR_VAD_END_SILENCE)
    parser.add_argument("--pre-roll", type=float, default=config.ASR_PRE_ROLL)
    parser.add_argument("--post-roll", type=float, default=config.ASR_POST_ROLL)
    parser.add_argument("--hotwords", default=config.ASR_HOTWORDS)
    parser.add_argument("--debug-audio", action="store_true", help="显示 VAD 音量和空结果诊断")
    args = parser.parse_args()
    if args.list_devices:
        selected_model = config.CHAT_MODEL
    else:
        try:
            selected_model = choose_chat_model(args.model)
        except (KeyboardInterrupt, EOFError):
            print("\n正在退出…")
            return 0
    if args.text:
        return run_text_mode(selected_model)
    return run_voice_mode(
        model=selected_model,
        model_dir=args.model_dir,
        vad_model_dir=args.vad_model_dir,
        device=args.device,
        threads=args.threads,
        vad_threshold=args.vad_threshold,
        vad_end_silence=args.vad_end_silence,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        hotwords=args.hotwords,
        debug_audio=args.debug_audio,
        list_devices=args.list_devices,
    )


if __name__ == "__main__":
    sys.exit(main())
