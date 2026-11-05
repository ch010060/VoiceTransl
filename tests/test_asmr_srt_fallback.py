"""Gate A.2: bounded ratio-only fallback, evidence integrity and readable warnings."""
from collections import Counter
from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import PlanRejected, build_execution_plan, validate_srt
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder, ProtocolError
from scripts.asmr_transcribe_lib.gui_results import (
    _digest, _verify_normalization, atomic_json, read_json, reconcile_run, resume_provenance, safe_ref,
)
from scripts.asmr_transcribe_lib.srt_normalization import (
    FALLBACK_POLICY, FALLBACK_WARNING_CODE, WARNING_CODE, TimestampOrderError, normalize_srt_bytes,
)
from scripts.asmr_transcribe_lib.srt_utils import format_srt, parse_srt
from test_asmr_gui_cleanup import prepare, intermediates
from test_asmr_gui_srt_normalization import RAW
from test_asmr_srt_normalization import fixture


def raw_case(count=45, backward=220, two=False):
    reversals = {10: backward, **({15: 10} if two else {})}
    return format_srt(fixture(max(count, 175), reversals)[:count]).encode()


@pytest.mark.parametrize("count,backward", [(45, 220), (20, 900), (20, 1000)])
@pytest.mark.parametrize("bom,newline", [(b"", b"\n"), (b"\xef\xbb\xbf", b"\r\n")])
def test_ratio_only_single_reversal_fallback_is_start_only(tmp_path, count, backward, bom, newline):
    raw = bom + raw_case(count, backward).replace(b"\n", newline)
    before = parse_srt(raw.decode())
    fixed, changes = normalize_srt_bytes(raw)
    after = parse_srt(fixed.decode())
    assert len(changes) == 1
    assert changes[0]["fallback"] == {"fallback_used": True, "reversal_count": 1,
                                       "max_backward_ms": backward, "normalization_policy": FALLBACK_POLICY}
    assert fixed == raw.replace(before[10].timestamp.encode(), after[10].timestamp.encode(), 1)
    assert [(b.index, b.text) for b in after] == [(b.index, b.text) for b in before]
    assert [b.timestamp.split(" --> ")[1] for b in after] == [b.timestamp.split(" --> ")[1] for b in before]
    assert after[:10] == before[:10] and after[11:] == before[11:]
    output = tmp_path / "normalized.srt"
    output.write_bytes(fixed)
    validate_srt(output)
    assert normalize_srt_bytes(fixed) == (fixed, [])


@pytest.mark.parametrize("raw,reason", [
    (raw_case(20, 1001), "displacement_limit"),
    (raw_case(45, two=True), "ratio_limit"),
    (raw_case(20).replace(b"00:01:29,780 --> 00:01:39,000", b"00:01:29,780 --> 00:01:29,999"), "repair_exceeds_end"),
], ids=["offset_limit", "two_reversals", "start_exceeds_end"])
def test_fallback_does_not_relax_other_guards(raw, reason):
    with pytest.raises(TimestampOrderError, match=reason):
        normalize_srt_bytes(raw)


@pytest.mark.parametrize("raw", [RAW, raw_case(50)])
def test_normal_policy_still_applies_without_fallback(raw):
    _, changes = normalize_srt_bytes(raw)
    assert len(changes) == 1 and "fallback" not in changes[0]


def test_correctly_ordered_short_srt_is_byte_stable():
    raw = format_srt(fixture(175, {})[:20]).encode()
    assert normalize_srt_bytes(raw) == (raw, [])


def execute_case(tmp_path, *, mode="full", shared=False, keep=False, fail_translation=False):
    names = ("work/normal/Track01.flac",)
    if shared:
        names += ("work/variant/Track01(SEcut).flac", "work/variant/Track01(SEcut).mp3")
    plan, store = prepare(tmp_path, mode=mode, keep=keep, names=names)
    calls = Counter()
    def asr(*, transcribe_srt_path, **kwargs):
        calls["asr"] += 1
        transcribe_srt_path.write_bytes(raw_case())
        return transcribe_srt_path
    def translate(source, target):
        calls["translation"] += 1
        if fail_translation:
            raise RuntimeError("synthetic translation failure")
        target.write_bytes(source.read_bytes())
        return target
    def convert(source, target):
        calls["opencc"] += 1
        target.write_bytes(source.read_bytes())
        return target
    outcomes = run_plan(plan, project_root=tmp_path, transcriber=asr,
                        translator=translate if mode == "full" else None, opencc_converter=convert,
                        observer=store.observer, normalization_store=store.record_normalization,
                        intermediate_store=store.intermediate_cleanup.capture)
    return plan, store, outcomes, store.finish(outcomes), calls


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_fallback_receipts_events_logs_dependencies_cleanup_and_resume(tmp_path, mode, shared, keep):
    plan, store, outcomes, result, calls = execute_case(tmp_path, mode=mode, shared=shared, keep=keep)
    owner = next(t for t in plan.tracks if t.task.action == "process")
    outcome = next(o for o in outcomes if o.track_id == owner.track_id)
    assert outcome.status == ("success" if mode == "full" else "transcribed")
    assert outcome.warnings == (WARNING_CODE, FALLBACK_WARNING_CODE)
    assert calls == ({"asr": 1, "translation": 1, "opencc": 1} if mode == "full" else {"asr": 1})
    assert result["counts"]["copied"] == (2 if shared else 0) and result["outcome"] == "completed"
    assert result["warnings"] == []
    rec = read_json(store.run_dir / "receipts" / (owner.track_id + ".json"))
    audit = read_json(safe_ref(store.run_dir, rec["timestamp_normalization"]["ref"]))
    assert rec["warnings"] == [WARNING_CODE, FALLBACK_WARNING_CODE]
    assert audit["fallback"] == {"fallback_used": True, "reversal_count": 1,
                                 "max_backward_ms": 220, "normalization_policy": FALLBACK_POLICY}
    assert safe_ref(store.run_dir, audit["raw_ref"]).read_bytes() == raw_case()
    validate_srt(safe_ref(store.run_dir, audit["normalized_ref"]))
    events = JSONLDecoder(store.run_id).feed((store.run_dir / "events.jsonl").read_bytes())
    terminal = next(e for e in events if e["event"] == "track_finished" and e["track_id"] == owner.track_id)
    assert terminal["payload"]["warnings"] == [WARNING_CODE, FALLBACK_WARNING_CODE]
    text = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert FALLBACK_WARNING_CODE in text and "單段小幅倒退已受限修復" in text and "Synthetic" not in text
    if mode == "full":
        assert all(p.exists() == keep for p in intermediates(owner))
        assert result["intermediate_cleanup"][owner.track_id]["status"] == ("disabled" if keep else "completed")
    else:
        assert owner.task.transcribe_srt_path.exists()
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    assert len(proofs) == len(plan.tracks)
    assert build_execution_plan(plan.root, mode=mode, settings={"keep_intermediate_files": keep}, provenance=proofs).executable


@pytest.mark.parametrize("field", ["fallback_used", "reversal_count", "max_backward_ms", "normalization_policy",
                                    "missing_fallback", "missing_warning", "typed_used", "typed_count"])
def test_rehashed_fallback_evidence_and_missing_warning_are_rejected(tmp_path, field):
    plan, store, _, _, _ = execute_case(tmp_path, mode="transcribe_only")
    rec = read_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))
    path = safe_ref(store.run_dir, rec["timestamp_normalization"]["ref"])
    audit = read_json(path)
    if field == "missing_warning":
        rec["warnings"].remove(FALLBACK_WARNING_CODE)
    else:
        if field == "missing_fallback":
            del audit["fallback"]
        elif field in {"typed_used", "typed_count"}:
            audit["fallback"]["fallback_used" if field == "typed_used" else "reversal_count"] = 1 if field == "typed_used" else 1.0
        else:
            audit["fallback"][field] = {"fallback_used": False, "reversal_count": 2,
                                         "max_backward_ms": 1001, "normalization_policy": "unknown"}[field]
        atomic_json(path, audit)
        rec["timestamp_normalization"]["hash"] = _digest(audit)
    with pytest.raises(ProtocolError):
        _verify_normalization(store.run_dir, rec)


def test_fallback_warning_without_audit_is_rejected(tmp_path):
    with pytest.raises(ProtocolError, match="lacks evidence"):
        _verify_normalization(tmp_path, {"warnings": [FALLBACK_WARNING_CODE]})


def test_fallback_warning_cannot_be_added_to_normal_policy_evidence(tmp_path):
    from test_asmr_gui_srt_normalization import setup, execute
    plan, store, _ = setup(tmp_path, "transcribe_only")
    execute(plan, store)
    rec = read_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))
    rec["warnings"].append(FALLBACK_WARNING_CODE)
    with pytest.raises(ProtocolError):
        _verify_normalization(store.run_dir, rec)


def test_applied_fallback_warning_survives_later_failure(tmp_path):
    plan, store, outcomes, result, _ = execute_case(tmp_path, fail_translation=True)
    assert outcomes[0].status == "failed" and result["counts"]["failed"] == 1
    rec = read_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))
    assert rec["warnings"] == [WARNING_CODE, FALLBACK_WARNING_CODE] and rec["provenance"] is None
    assert plan.tracks[0].task.transcribe_srt_path.exists()
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


def test_fallback_does_not_adopt_preexisting_unverified_transcript(tmp_path):
    plan, store = prepare(tmp_path)
    store._readable.close(wait=True)
    path = plan.tracks[0].task.transcribe_srt_path
    path.write_bytes(raw_case())
    legacy = build_execution_plan(plan.root)
    with pytest.raises(PlanRejected, match="UNVERIFIED_LEGACY"):
        legacy.assert_ready()
    assert path.read_bytes() == raw_case()


def test_fallback_requires_strict_validation_before_publication(tmp_path, monkeypatch):
    from scripts.asmr_transcribe_lib import gui_execution
    from test_asmr_gui_srt_normalization import setup, execute
    plan, store, _ = setup(tmp_path)
    original = gui_execution.validate_srt
    checked = []
    def strict(path):
        if Path(path).name.endswith(".tmp"):
            checked.append(path)
            raise ValueError("synthetic strict validation rejection")
        return original(path)
    monkeypatch.setattr(gui_execution, "validate_srt", strict)
    calls = Counter()
    outcomes, _ = execute(plan, store, raw=raw_case(), calls=calls)
    assert checked and outcomes[0].status == "failed" and calls == {"asr": 1}
    assert plan.tracks[0].task.transcribe_srt_path.read_bytes() == raw_case()
    assert not plan.tracks[0].task.final_srt_path.exists()
