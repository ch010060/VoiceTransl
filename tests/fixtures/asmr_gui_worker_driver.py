"""Test-only adapter injection; production worker has no fake-backend switch."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asmr_gui_worker import main


def fake_asr(*, audio_path, transcribe_srt_path, **kwargs):
    # Both Python and native fd output contain private sentinel material. The
    # worker must prevent all of it from entering either public stream.
    print("PRIVATE_CAPTION_SENTINEL token=PRIVATE_SECRET")
    os.write(1, b"NATIVE_PRIVATE_CAPTION_SENTINEL\n")
    os.write(2, b"NATIVE_PRIVATE_SECRET_SENTINEL\n")
    subprocess.run([sys.executable, "-c", "import os; os.write(1,b'CHILD_PRIVATE_SENTINEL\\n'); os.write(2,b'CHILD_PRIVATE_SECRET\\n')"], check=True)
    if "slow" in audio_path.stem:
        time.sleep(.7)
    callback = kwargs.get("transcription_progress_callback")
    if callback:
        callback(1., 10., 10.)
    if "fail" in audio_path.stem:
        raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")
    transcribe_srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nPRIVATE_SUBTITLE_TEXT\n", encoding="utf-8")
    return transcribe_srt_path


def fake_copy(source, target):
    shutil.copyfile(source, target)
    return target


mode = sys.argv[1]
if mode == "native_exit7":
    # Exercise the production strict adapter against an actual owned child
    # process, with a synthetic SRT and exit7. No model or executable download.
    import scripts.asmr_gui_worker as worker
    from scripts.asmr_transcribe_lib import local_faster_whisper as native
    worker.preflight = lambda *args: {"available": True, "inference_proven": False}
    native.preflight_faster_whisper = lambda *args: Path(sys.executable)
    native.build_faster_whisper_argv = lambda exe, audio, output, config: [sys.executable, "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('1\\n00:00:00,000 --> 00:00:01,000\\nPRIVATE_NATIVE_TEXT\\n',encoding='utf-8'); print('PRIVATE_NATIVE_CHILD_LOG'); sys.exit(7)",
        str(output / (audio.stem + ".srt"))]
    raise SystemExit(main(sys.argv[2:]))
adapters = {"preflight": lambda *args: {"available": mode != "preflight_fail", "inference_proven": False},
            "transcriber": fake_asr, "translator": None if mode == "transcribe_only" else fake_copy,
            "opencc_converter": fake_copy}
raise SystemExit(main(sys.argv[2:], adapters=adapters))
