"""Bounded asynchronous curated GUI diagnostics; event authority stays in RunStore."""
from pathlib import Path
import re
from scripts.asmr_transcribe_lib.gui_logging import DiagnosticLog


def plan_rejection_diagnostic(intent, rejection):
    """Full local path evidence, but only bounded curated facts for the GUI.

    Never serialize settings, exception text/tracebacks or subtitle contents.
    This is presentation only: the caller still rejects the execution plan.
    """
    catalog = intent.catalog
    tracks = catalog.track_index if catalog is not None else {}
    path_tracks = {}
    for track in tracks.values():
        task = track.task
        for path in (task.audio_path, task.final_srt_path, task.transcribe_srt_path,
                     task.translate_srt_path, task.opencc_srt_path, task.existing_subtitle_path):
            if path is not None:
                for variant in (path, path.with_name(path.name + ".tmp")):
                    path_tracks.setdefault(variant, set()).add(track.track_id)
    grouped, affected = {}, set()
    for issue in rejection.issues:
        # Issue codes are curated machine categories, never arbitrary messages.
        code = issue.code if re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", issue.code) else "INVALID_ISSUE_CODE"
        group = grouped.setdefault(code, {"issues": 0, "tracks": set(), "paths": []})
        group["issues"] += 1
        ids = set(issue.track_ids)
        if not ids:
            for path in issue.paths:
                ids.update(path_tracks.get(path, ()))
        group["tracks"].update(ids)
        affected.update(ids)
        for path in issue.paths:
            if len(group["paths"]) < 5 and path not in group["paths"]:
                group["paths"].append(path)
    origin = getattr(rejection, "diagnostic_origin", "execution_validation")
    if origin not in {"catalog_freshness_validation", "plan_construction", "start_freshness_validation"}:
        origin = "execution_validation"
    summary = [f"origin={origin} issues={len(rejection.issues)} tracks={len(affected)}"]
    for code, group in list(grouped.items())[:8]:
        # Filename samples avoid exposing the private library root. Full paths
        # and Track identities are available only in the local JSON artifact.
        samples = ["".join(c if c.isprintable() else "?" for c in p.name)[:80] for p in group["paths"]]
        summary.append(f"{code}: tracks={len(group['tracks'])} issues={group['issues']}"
                       + (" paths=" + ", ".join(samples) if samples else ""))
    if len(grouped) > 8:
        summary.append(f"additional_codes={len(grouped) - 8}")
    report = {"schema": 1, "operation_id": intent.operation_id, "origin": origin,
              "root": str(intent.root), "mode": intent.mode, "limit": intent.limit,
              "selected_work_ids": list(intent.selected),
              "catalog_plan_hash": catalog.plan_hash if catalog is not None else None,
              "settings_hash": catalog.settings_hash if catalog is not None else None,
              "affected_track_count": len(affected),
              "issues": [{"code": i.code, "track_ids": list(i.track_ids),
                          "paths": [str(p) for p in i.paths], "detail": i.detail}
                         for i in rejection.issues],
              "tracks": [{"track_id": key, "audio_path": str(tracks[key].task.audio_path),
                          "work_id": tracks[key].work_id,
                          "action": tracks[key].task.action,
                          "dependencies": list(tracks[key].dependencies)}
                         for key in sorted(affected) if key in tracks]}
    return report, " · ".join(summary)
