from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
    authenticate_compatibility_resolution,
    load_compatibility_resolution,
    load_family_selection,
)
from experiments.g3_pretrained_item_embeddings.launchers.native500m import PROJECT_ROOT


_RQ_FAMILIES = {
    1: ("untied_control", "rq1_content_input"),
    2: ("rq2_content_concat",),
    3: (
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
    ),
    4: ("rq4_artist", "rq4_album", "rq4_artist_album"),
    5: ("rq5_global_gate", "rq5_frequency_gate"),
}
_QUESTIONS = {
    1: "What happens when pretrained embeddings replace item IDs?",
    2: "Does concatenating content and item ID help?",
    3: "Which prediction embedding is best?",
    4: "Do artist and album features help?",
    5: "Does a frequency-adaptive content gate help?",
}
_LABELS = {
    "baseline": "two-layer G1-best tied baseline",
    "untied_control": "untied learned-ID control (secondary mechanism control)",
    "rq1_content_input": "normalized frozen content",
    "rq2_content_concat": "item ID + normalized content",
    "rq3_output_learned": "learned item output",
    "rq3_output_frozen_content": "frozen-content output",
    "rq3_output_trainable_content": "trainable-content output",
    "rq3_output_learned_frozen_content": "learned ID + frozen-content output",
    "rq3_output_learned_trainable_content": "learned ID + trainable-content output",
    "rq4_artist": "artist metadata",
    "rq4_album": "album metadata",
    "rq4_artist_album": "artist + album metadata",
    "rq5_global_gate": "learned global content gate",
    "rq5_frequency_gate": "frequency-adaptive content gate",
    "aggregate": "best compatible combination",
}
_METHOD_DESCRIPTIONS = {
    "baseline": "the frozen two-layer G1-best model with one tied learned item-ID table",
    "untied_control": "separate learned history and catalog ID tables isolate the tying change",
    "rq1_content_input": "normalized frozen content replaces learned IDs in history only",
    "rq2_content_concat": "learned IDs and normalized frozen content are concatenated and encoded by DenseNet",
    "rq3_output_learned": "an independent learned item-ID table provides catalog targets",
    "rq3_output_frozen_content": "normalized frozen content is projected into catalog-target space",
    "rq3_output_trainable_content": "pretrained content is normalized and fine-tuned as the catalog target",
    "rq3_output_learned_frozen_content": "learned IDs and normalized frozen content form the catalog target",
    "rq3_output_learned_trainable_content": "learned IDs and normalized trainable content form the catalog target",
    "rq4_artist": "train-only artist features are pooled and fused with the tied item representation",
    "rq4_album": "train-only album features are pooled and fused with the tied item representation",
    "rq4_artist_album": "artist and album features are pooled separately and fused together",
    "rq5_global_gate": "one learned scalar controls the content contribution to fixed concatenation",
    "rq5_frequency_gate": "an FP32 frequency-conditioned gate controls the content contribution",
}
_METRICS = ("recall@100", "ndcg@100")
_NATIVE500M_NOISE_SOURCE_SHA256 = (
    "01d5cd24f599afe01737676e4bd71b04c4328fe389d47b8395b8f5f6f27a926d"
)


@dataclass(frozen=True)
class RenderedNative500MReports:
    reader: str
    scratchpad: str


def render_native500m_reports(
    *,
    selection_paths: Mapping[str, Path],
    conclusions: Mapping[int, Sequence[str]],
    compatibility_resolution_path: Path | None = None,
) -> RenderedNative500MReports:
    expected = {
        "baseline",
        *(family for families in _RQ_FAMILIES.values() for family in families),
    }
    required = (
        expected
        if compatibility_resolution_path is not None
        else {
            *expected,
            "aggregate",
        }
    )
    if set(selection_paths) != required:
        raise ValueError("native-500M report requires every approved family selection")
    if set(conclusions) != set(_RQ_FAMILIES) or any(
        not 3 <= len(lines) <= 5
        or any(not line.strip() or "\n" in line for line in lines)
        for lines in conclusions.values()
    ):
        raise ValueError("each RQ requires three to five hand-written conclusion lines")
    selections = {
        family: load_family_selection(path) for family, path in selection_paths.items()
    }
    if any(
        document.get("family_id") != family for family, document in selections.items()
    ):
        raise ValueError("report selection family identity differs")
    baseline = selections["baseline"]["winner"]
    relative_bands = _relative_noise_bands()
    thresholds = {
        metric: float(baseline["metrics"][metric]) * relative_bands[metric]
        for metric in _METRICS
    }
    baseline_tail = _tail_recall(baseline)
    if baseline_tail is None:
        raise ValueError("baseline selection has no authenticated tail Recall@100")
    tail_threshold = baseline_tail * relative_bands["recall@100"]
    compatibility_states = None
    if compatibility_resolution_path is not None:
        compatibility_states, authenticated_final = (
            _authenticate_report_compatibility_chain(compatibility_resolution_path)
        )
        _validate_report_compatibility_closure(
            states=compatibility_states,
            authenticated_final=authenticated_final,
            selection_paths=selection_paths,
            selections=selections,
            recall_threshold=thresholds["recall@100"],
            tail_threshold=tail_threshold,
        )
    opening = (
        "Native Yambda-500M full-user evidence. Operational thresholds reuse the "
        "reviewed native-500M relative dispersion and scale this experiment's "
        "baseline: "
        + ", ".join(f"{metric}={thresholds[metric]:.6f}" for metric in _METRICS)
        + f"; tail Recall@100 proxy={tail_threshold:.6f}."
    )
    reader = ["# G3: pretrained item embeddings", "", opening]
    scratchpad = ["# G3: pretrained item embeddings native-500M results"]
    for number, families in _RQ_FAMILIES.items():
        documents = [selections[family] for family in families]
        displayed_families = families
        if number == 5:
            displayed_families = ("rq2_content_concat", *families)
            documents = [selections[family] for family in displayed_families]
        rows = [baseline, *(document["winner"] for document in documents)]
        winner_index = _approved_winner_index(
            number,
            rows,
            ("baseline", *displayed_families),
            thresholds,
            tail_threshold,
        )
        table = _comparison_table(
            rows,
            ("baseline", *displayed_families),
            winner_index=winner_index,
            thresholds=thresholds,
        )
        item_slices = _slice_table(
            rows,
            ("baseline", *displayed_families),
            axis="item_frequency",
            names=("head", "mid", "tail"),
            winner_index=winner_index,
        )
        history_slices = _slice_table(
            rows,
            ("baseline", *displayed_families),
            axis="user_history",
            names=("low", "mid", "high"),
            winner_index=winner_index,
        )
        heading = f"RQ{number}: {_QUESTIONS[number]}"
        descriptions = _method_descriptions(("baseline", *displayed_families))
        reader.extend(
            (
                "",
                f"## {heading}",
                "",
                descriptions,
                "",
                table,
                "",
                "Item-frequency Recall@100 (descriptive; no slice-specific repeat band):",
                "",
                item_slices,
                "",
                "History-length Recall@100 (descriptive; no slice-specific repeat band):",
                "",
                history_slices,
                "",
                "  \n".join(conclusions[number]),
            )
        )
        scratchpad.extend(
            ("", f"## {heading}", "", table, "", item_slices, "", history_slices)
        )
    if compatibility_resolution_path is None:
        aggregate_table = _aggregate_table(
            baseline=baseline,
            aggregate_selection=selections["aggregate"],
            thresholds=thresholds,
        )
        aggregate_note = (
            "The table uses the trained compatible aggregate and authenticated "
            "non-overlapping predecessor/bridge chain."
        )
    else:
        assert compatibility_states is not None
        resolution = compatibility_states[-1]
        if resolution["next_conditional_family"] is not None:
            raise ValueError(
                "compatibility closure still authorizes a conditional family"
            )
        aggregate_table = _compatibility_aggregate_table(
            baseline=baseline,
            thresholds=thresholds,
            states=compatibility_states,
        )
        most_reference = resolution["most_specific_selection"]
        most = load_family_selection(PROJECT_ROOT / str(most_reference["path"]))[
            "winner"
        ]
        if most["row_id"] != most_reference["row_id"]:
            raise ValueError("compatibility final selection row differs")
        aggregate_note = (
            "No compatible treatment qualified, so the aggregate reuses the baseline without a duplicate run."
            if most["job"]["family_id"] == "baseline"
            else "The most-specific authenticated standalone or natural bridge already contains every surviving compatible component and is reused as the aggregate."
        )
    reader.extend(
        (
            "",
            "## Aggregated improvement",
            "",
            aggregate_table,
            "",
            aggregate_note,
        )
    )
    scratchpad.extend(("", "## Aggregated improvement", "", aggregate_table))
    return RenderedNative500MReports(
        reader="\n".join(reader) + "\n",
        scratchpad="\n".join(scratchpad) + "\n",
    )


def _reused_aggregate_table(
    baseline: Mapping[str, object], reused: Mapping[str, object]
) -> str:
    lines = [
        "| metric | baseline | aggregate | gain | gain (%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for metric in _METRICS:
        base = float(baseline["metrics"][metric])
        value = float(reused["metrics"][metric])
        gain = value - base
        lines.append(
            f"| {metric} | {base:.6f} | {value:.6f} | {gain:+.6f} | {100 * gain / base:+.2f}% |"
        )
    return "\n".join(lines)


def _authenticate_report_compatibility_chain(
    final_state_path: Path,
) -> tuple[tuple[Mapping[str, object], ...], object]:
    root = PROJECT_ROOT.resolve(strict=True)
    path = final_state_path
    states = []
    expected_reference = None
    final_authenticated = None
    seen = set()
    while True:
        document, authenticated = authenticate_compatibility_resolution(path, root=root)
        if final_authenticated is None:
            final_authenticated = authenticated
        relative_path = str(authenticated.relative_path)
        if relative_path in seen:
            raise ValueError("compatibility closure contains a state cycle")
        seen.add(relative_path)
        if int(document["generation"]) != int(authenticated.generation):
            raise ValueError("compatibility closure generation identity differs")
        if expected_reference is not None and _file_reference_identity(
            expected_reference
        ) != _authenticated_state_identity(authenticated):
            raise ValueError("compatibility closure prior state identity differs")
        states.append(document)
        expected_reference = document.get("prior_state")
        if expected_reference is None:
            break
        path = root / str(expected_reference["path"])
    states.reverse()
    if [state["generation"] for state in states] != list(range(len(states))):
        raise ValueError("compatibility closure generation chain differs")
    assert final_authenticated is not None
    return tuple(states), final_authenticated


def _validate_report_compatibility_closure(
    *,
    states: Sequence[Mapping[str, object]],
    authenticated_final: object,
    selection_paths: Mapping[str, Path],
    selections: Mapping[str, Mapping[str, object]],
    recall_threshold: float,
    tail_threshold: float,
) -> None:
    expected_references = {
        family: _caller_selection_identity(selection_paths[family], document)
        for family, document in selections.items()
    }
    expected_thresholds = {
        "recall@100": recall_threshold,
        "tail_recall@100": tail_threshold,
    }
    for state in states:
        references = state.get("standalone_selections")
        if not isinstance(references, dict) or set(references) != set(
            expected_references
        ):
            raise ValueError("compatibility closure standalone selections differ")
        for family, expected in expected_references.items():
            reference = references[family]
            if (
                not isinstance(reference, dict)
                or set(reference)
                != {
                    "role",
                    "path",
                    "size_bytes",
                    "sha256",
                    "logical_sha256",
                    "row_id",
                }
                or reference.get("role") != "family_evidence"
                or _row_selection_identity(reference) != expected
            ):
                raise ValueError(
                    f"compatibility closure standalone selection identity differs: {family}"
                )
        if state.get("gate_thresholds") != expected_thresholds:
            raise ValueError("compatibility closure threshold source differs")
    final_reference = states[-1].get("most_specific_selection")
    if not isinstance(final_reference, dict) or _row_selection_identity(
        final_reference
    ) != _authenticated_final_selection_identity(authenticated_final):
        raise ValueError("compatibility closure final selection identity differs")


def _caller_selection_identity(
    path: Path, document: Mapping[str, object]
) -> tuple[object, ...]:
    root = PROJECT_ROOT.resolve(strict=True)
    if path.is_symlink():
        raise ValueError("report selection is not a regular project artifact")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("report selection is not a regular project artifact")
    return (
        resolved.relative_to(root).as_posix(),
        resolved.stat().st_size,
        hashlib.sha256(resolved.read_bytes()).hexdigest(),
        document["sha256"],
        document["winner"]["row_id"],
    )


def _file_reference_identity(reference: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        reference.get(field)
        for field in ("path", "size_bytes", "sha256", "logical_sha256")
    )


def _row_selection_identity(reference: Mapping[str, object]) -> tuple[object, ...]:
    return _file_reference_identity(reference) + (reference.get("row_id"),)


def _authenticated_state_identity(authenticated: object) -> tuple[object, ...]:
    return (
        authenticated.relative_path,
        authenticated.size_bytes,
        authenticated.physical_sha256,
        authenticated.logical_sha256,
    )


def _authenticated_final_selection_identity(
    authenticated: object,
) -> tuple[object, ...]:
    return (
        authenticated.most_specific_selection_path,
        authenticated.most_specific_selection_size_bytes,
        authenticated.most_specific_selection_physical_sha256,
        authenticated.most_specific_selection_logical_sha256,
        authenticated.most_specific_selected.coordinate.source_id,
    )


def _compatibility_aggregate_table(
    *,
    baseline: Mapping[str, object],
    thresholds: Mapping[str, float],
    final_state_path: Path | None = None,
    states: Sequence[Mapping[str, object]] | None = None,
) -> str:
    if states is None:
        if final_state_path is None:
            raise ValueError("compatibility arithmetic requires one state chain")
        loaded_states = []
        path = final_state_path
        while True:
            state = load_compatibility_resolution(path)
            loaded_states.append(state)
            prior = state["prior_state"]
            if prior is None:
                break
            path = PROJECT_ROOT / str(prior["path"])
        loaded_states.reverse()
        states = loaded_states
    elif final_state_path is not None:
        raise ValueError("compatibility arithmetic received mixed state sources")
    initial = states[0]

    def winner(reference: Mapping[str, object]) -> Mapping[str, object]:
        document = load_family_selection(PROJECT_ROOT / str(reference["path"]))
        if document["winner"]["row_id"] != reference["row_id"]:
            raise ValueError("compatibility arithmetic row differs")
        return document["winner"]

    component_gains = {metric: 0.0 for metric in _METRICS}
    input_row = winner(initial["component_targets"]["input"])
    for metric in _METRICS:
        component_gains[metric] += float(input_row["metrics"][metric]) - float(
            baseline["metrics"][metric]
        )
    if initial["included"]["output"] is not None:
        output = winner(initial["included"]["output"])
        learned_reference = initial["standalone_selections"]["rq3_output_learned"]
        learned = load_family_selection(PROJECT_ROOT / str(learned_reference["path"]))[
            "winner"
        ]
        for metric in _METRICS:
            component_gains[metric] += float(output["metrics"][metric]) - float(
                learned["metrics"][metric]
            )
    if initial["included"]["metadata"] is not None:
        metadata = winner(initial["included"]["metadata"])
        for metric in _METRICS:
            component_gains[metric] += float(metadata["metrics"][metric]) - float(
                baseline["metrics"][metric]
            )
    for state in states[1:]:
        transition = state["completed_transition"]
        if transition["decision"] != "accept":
            continue
        selected = winner(transition["selected_selection"])
        predecessor = winner(transition["predecessor_reference"])
        for metric in _METRICS:
            component_gains[metric] += float(selected["metrics"][metric]) - float(
                predecessor["metrics"][metric]
            )
    aggregate = winner(states[-1]["most_specific_selection"])
    rendered = [
        "| metric | baseline | trained aggregate | gain | gain (%) | standalone + bridge gain | interaction gap | resolution band | classification |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for metric in _METRICS:
        base = float(baseline["metrics"][metric])
        value = float(aggregate["metrics"][metric])
        gain = value - base
        gap = gain - component_gains[metric]
        band = thresholds[metric]
        classification = (
            "positive" if gap > band else "negative" if gap < -band else "unresolved"
        )
        rendered.append(
            f"| {metric} | {base:.6f} | {value:.6f} | {gain:+.6f} | "
            f"{100 * gain / base:+.2f}% | {component_gains[metric]:+.6f} | "
            f"{gap:+.6f} | {band:.6f} | {classification} |"
        )
    return "\n".join(rendered)


def _validation_winner_index(
    rows: Sequence[Mapping[str, object]],
    eligible_indices: Sequence[int] | None = None,
) -> int:
    indices = range(len(rows)) if eligible_indices is None else eligible_indices
    return min(
        indices,
        key=lambda index: (
            -float(rows[index]["selection_metrics"]["recall@100"]),
            -float(rows[index]["selection_metrics"]["ndcg@100"]),
            int(rows[index]["job"]["manifest_order"]),
        ),
    )


def _approved_winner_index(
    number: int,
    rows: Sequence[Mapping[str, object]],
    families: Sequence[str],
    thresholds: Mapping[str, float],
    tail_threshold: float,
) -> int:
    band = thresholds["recall@100"]
    recall = [float(row["metrics"]["recall@100"]) for row in rows]
    if number == 1:
        content = families.index("rq1_content_input")
        if recall[content] > recall[0] + band or (
            recall[content] >= recall[0] - band
            and _tail_recall(rows[content]) is not None
            and _tail_recall(rows[0]) is not None
            and _tail_recall(rows[content]) > _tail_recall(rows[0]) + tail_threshold
        ):
            return content
        return 0
    if number == 2:
        return 1 if recall[1] > recall[0] + band else 0
    if number == 3:
        selected = _validation_winner_index(rows, tuple(range(1, len(rows))))
        learned = families.index("rq3_output_learned")
        return (
            selected
            if selected == learned or recall[selected] > recall[learned] + band
            else learned
        )
    if number == 4:
        selected = _validation_winner_index(rows, tuple(range(1, len(rows))))
        if recall[selected] > recall[0] + band or (
            recall[selected] >= recall[0] - band
            and _tail_recall(rows[selected]) is not None
            and _tail_recall(rows[0]) is not None
            and _tail_recall(rows[selected]) > _tail_recall(rows[0]) + tail_threshold
        ):
            return selected
        return 0
    if number == 5:
        fixed = families.index("rq2_content_concat")
        global_gate = families.index("rq5_global_gate")
        frequency = families.index("rq5_frequency_gate")
        fixed_qualifies = recall[fixed] > recall[0] + band
        global_qualifies = (
            recall[global_gate] > recall[fixed] + band
            and recall[global_gate] > recall[0] + band
        )
        frequency_tail = _tail_recall(rows[frequency])
        comparator_tails = [
            _tail_recall(rows[index]) for index in (0, fixed, global_gate)
        ]
        frequency_qualifies = (
            all(
                recall[frequency] >= recall[index] - band
                for index in (0, fixed, global_gate)
            )
            and frequency_tail is not None
            and all(
                value is not None and frequency_tail > value + tail_threshold
                for value in comparator_tails
            )
        )
        eligible = [
            index
            for index, qualifies in (
                (fixed, fixed_qualifies),
                (global_gate, global_qualifies),
                (frequency, frequency_qualifies),
            )
            if qualifies
        ]
        return 0 if not eligible else _band_aware_choice(rows, eligible, band)
    raise ValueError(f"unsupported research question {number}")


def _band_aware_choice(
    rows: Sequence[Mapping[str, object]], indices: Sequence[int], band: float
) -> int:
    leader = min(
        indices,
        key=lambda index: (
            -float(rows[index]["metrics"]["recall@100"]),
            int(rows[index]["job"]["manifest_order"]),
        ),
    )
    leader_recall = float(rows[leader]["metrics"]["recall@100"])
    unresolved = [
        index
        for index in indices
        if leader_recall - float(rows[index]["metrics"]["recall@100"]) <= band
    ]
    if len(unresolved) == 1:
        return leader
    return min(
        unresolved,
        key=lambda index: (
            -float(_tail_recall(rows[index]) or 0.0),
            -float(rows[index]["metrics"]["ndcg@100"]),
            int(rows[index]["job"]["manifest_order"]),
        ),
    )


def _comparison_table(
    rows: Sequence[Mapping[str, object]],
    families: Sequence[str],
    *,
    winner_index: int,
    thresholds: Mapping[str, float],
) -> str:
    baseline_metrics = rows[0]["metrics"]
    rendered = [
        "| variant | recall@100 | ndcg@100 |",
        "| :--- | :---: | :---: |",
    ]
    for index, (family, row) in enumerate(zip(families, rows, strict=True)):
        label = _LABELS[family]
        if index == winner_index:
            label = f"**{label}**"
        cells = [label]
        for metric in _METRICS:
            value = float(row["metrics"][metric])
            baseline = float(baseline_metrics[metric])
            if index == 0:
                cells.append(f"{value:.3f}")
            else:
                delta = value - baseline
                percent = 100 * delta / baseline
                cell = f"{percent:+.1f}% ({value:.3f})"
                if abs(delta) > thresholds[metric]:
                    color = "green" if delta > 0 else "red"
                    cell = f'<span style="color: {color}">{cell}</span>'
                cells.append(cell)
        rendered.append(f"| {' | '.join(cells)} |")
    return "\n".join(rendered)


def _slice_table(
    rows: Sequence[Mapping[str, object]],
    families: Sequence[str],
    *,
    axis: str,
    names: Sequence[str],
    winner_index: int,
) -> str:
    values = [[_slice_recall(row, axis, name) for name in names] for row in rows]
    rendered = [
        f"| variant | {' | '.join(names)} |",
        f"| :--- | {' | '.join(':---:' for _ in names)} |",
    ]
    for index, family in enumerate(families):
        label = _LABELS[family]
        if index == winner_index:
            label = f"**{label}**"
        cells = [label]
        for column, value in enumerate(values[index]):
            if index == 0:
                cells.append(f"{value:.3f}")
            else:
                baseline = values[0][column]
                cells.append(
                    f"{100 * (value - baseline) / baseline:+.1f}% ({value:.3f})"
                )
        rendered.append(f"| {' | '.join(cells)} |")
    return "\n".join(rendered)


def _slice_recall(row: Mapping[str, object], axis: str, name: str) -> float:
    slices = row.get("slices")
    if not isinstance(slices, dict) or not isinstance(slices.get(axis), dict):
        raise ValueError("selected evidence has no authenticated slice metrics")
    entry = slices[axis].get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
        raise ValueError("selected evidence has incomplete authenticated slices")
    value = float(entry["metrics"].get("recall@100"))
    if not math.isfinite(value) or value < 0:
        raise ValueError("selected evidence has an invalid slice metric")
    return value


def _method_descriptions(families: Sequence[str]) -> str:
    return "\n".join(
        f"- {_LABELS[family]} — {_METHOD_DESCRIPTIONS[family]}." for family in families
    )


def _tail_recall(row: Mapping[str, object]) -> float | None:
    slices = row.get("slices")
    if not isinstance(slices, dict):
        return None
    frequency = slices.get("item_frequency")
    if not isinstance(frequency, dict):
        return None
    tail = frequency.get("tail")
    if not isinstance(tail, dict) or not isinstance(tail.get("metrics"), dict):
        return None
    if "recall@100" not in tail["metrics"]:
        return None
    value = float(tail["metrics"]["recall@100"])
    return value if math.isfinite(value) else None


def _aggregate_table(
    *,
    baseline: Mapping[str, object],
    aggregate_selection: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> str:
    aggregate = aggregate_selection["winner"]
    component_gains = _non_overlapping_component_gains(aggregate_selection, baseline)
    rendered = [
        "| metric | baseline | trained aggregate | gain | gain (%) | standalone + bridge gain | interaction gap | resolution band | classification |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for metric in _METRICS:
        baseline_value = float(baseline["metrics"][metric])
        aggregate_value = float(aggregate["metrics"][metric])
        gain = aggregate_value - baseline_value
        component_gain = component_gains[metric]
        gap = gain - component_gain
        band = thresholds[metric]
        classification = (
            "positive" if gap > band else "negative" if gap < -band else "unresolved"
        )
        rendered.append(
            "| "
            + " | ".join(
                (
                    metric,
                    f"{baseline_value:.6f}",
                    f"{aggregate_value:.6f}",
                    f"{gain:+.6f}",
                    f"{100 * gain / baseline_value:+.2f}%",
                    f"{component_gain:+.6f}",
                    f"{gap:+.6f}",
                    f"{band:.6f}",
                    classification,
                )
            )
            + " |"
        )
    return "\n".join(rendered)


def _non_overlapping_component_gains(
    aggregate_selection: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, float]:
    references = aggregate_selection["winner"]["job"]["predecessor_artifacts"]
    allowed_roles = {"aggregate_input", "output_bridge", "metadata_bridge"}
    if (
        not isinstance(references, list)
        or not references
        or any(reference.get("role") not in allowed_roles for reference in references)
        or len({reference["role"] for reference in references}) != len(references)
    ):
        raise ValueError("aggregate selection has no authenticated arithmetic chain")
    gains = {metric: 0.0 for metric in _METRICS}
    for reference in references:
        selection = load_family_selection(PROJECT_ROOT / str(reference["path"]))
        winner = selection["winner"]
        if winner["row_id"] != reference["row_id"]:
            raise ValueError("aggregate arithmetic predecessor winner differs")
        if reference["role"] == "aggregate_input":
            predecessor = baseline
        else:
            predecessor_reference = selection.get("predecessor")
            if not isinstance(predecessor_reference, dict):
                raise ValueError("bridge selection has no authenticated predecessor")
            predecessor = load_family_selection(
                PROJECT_ROOT / str(predecessor_reference["path"])
            )["winner"]
        for metric in _METRICS:
            gains[metric] += float(winner["metrics"][metric]) - float(
                predecessor["metrics"][metric]
            )
    return gains


def _relative_noise_bands() -> dict[str, float]:
    path = (
        PROJECT_ROOT
        / "experiments/g1_sasrec_item_ids_likes/scratchpad/baseline_spread_500m.json"
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != _NATIVE500M_NOISE_SOURCE_SHA256:
        raise ValueError("native-500M noise calibration artifact differs")
    document = json.loads(path.read_text())
    if document.get("n") != 10 or set(document.get("seeds", [])) != set(range(10)):
        raise ValueError("native-500M noise calibration identity differs")
    result = {}
    for metric in _METRICS:
        value = float(document["metrics"][metric]["stddev_percent_of_mean"]) / 100
        if not math.isfinite(value) or value <= 0:
            raise ValueError("native-500M relative noise band is invalid")
        result[metric] = value
    return result
