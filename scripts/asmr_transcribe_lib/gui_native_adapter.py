"""GUI-only strict native exit evidence over the existing XXL adapter."""
from __future__ import annotations

from types import SimpleNamespace

from . import local_faster_whisper as native


def strict_xxl_transcriber(plan, metadata):
    by_audio = {track.task.audio_path.resolve(): track.track_id for track in plan.planned_tracks}

    def transcribe(*, audio_path, transcribe_srt_path, project_root, config,
                   transcription_progress_callback=None):
        track_id = by_audio[audio_path.resolve()]
        def run(argv, **options):
            # The legacy adapter's progress branch hides the return code from
            # callers. Its injectable runner lets this opt-in boundary retain
            # that evidence while reusing native parsing/output handling.
            code, _private_log = native._run_with_progress(
                argv, options["cwd"], transcription_progress_callback or (lambda *args: None))
            known = isinstance(code, int) and not isinstance(code, bool)
            metadata[track_id] = {"backend": "xxl", "native_exit_code": code if known else None,
                                  "returncode_known": known}
            return SimpleNamespace(returncode=code, stdout="", stderr="")
        output = native.run_faster_whisper_transcription(
            audio_path=audio_path, transcribe_srt_path=transcribe_srt_path,
            project_root=project_root, config=config, subprocess_run=run)
        evidence = metadata.get(track_id, {})
        if evidence.get("returncode_known") is not True or evidence.get("native_exit_code") != 0:
            # Keep the partial file, but do not certify it or continue translation.
            raise RuntimeError("ASR_NATIVE_EXIT_UNSUCCESSFUL")
        return output
    return transcribe
