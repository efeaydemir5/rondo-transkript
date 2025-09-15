#!/usr/bin/env python3
"""Generate linear transcription artefacts from HAHA.musicxml (WIP).

Currently extracts the first 20 linear measures (or fewer if file shorter) and emits:
 - transcript_full.txt (partial)
 - transcript_full.json (partial)
 - transcript_full.abc (partial)
 - measure_map.md (partial mapping table)

When complete this script will:
  1. Parse MusicXML (ElementTree) building measure objects
  2. Resolve repeats / endings into linear order
  3. Merge ties, mark slurs, articulations, dynamics, grace
  4. Output full files

Run:
    python scripts/generate_transcript.py HAHA.musicxml
"""
from __future__ import annotations
import sys, json, datetime, pathlib, xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

# --- Configuration constants ---
TURKISH_MAP = {"C":"Do","D":"Re","E":"Mi","F":"Fa","G":"Sol","A":"La","B":"Si"}

@dataclass
class Articulation:
    staccato: bool = False
    accent: bool = False
    tenuto: bool = False
    marcato: bool = False

    def tokens(self) -> List[str]:
        order = []
        if self.staccato: order.append("stacc")
        if self.accent: order.append("accent")
        if self.tenuto: order.append("tenuto")
        if self.marcato: order.append("marcato")
        return order

@dataclass
class NoteEvent:
    t: int          # onset in sixteenth units
    dur: int        # duration in sixteenth units (0 for grace)
    notes: List[str]
    art: List[str] = field(default_factory=list)
    dyn: Optional[str] = None
    slur_start: bool = False
    slur_end: bool = False
    grace: bool = False

@dataclass
class LinearMeasure:
    linear_index: int
    original_measure: int
    RH: List[NoteEvent] = field(default_factory=list)
    LH: List[NoteEvent] = field(default_factory=list)
    new_dynamic: Optional[str] = None
    tempo: Optional[int] = None

# --- Helper functions ---

def pitch_to_turkish(step: str, alter: int, octave: int) -> str:
    name = TURKISH_MAP[step]
    if alter == 1:
        name += '#'
    elif alter == -1:
        name += 'b'
    return f"{name}{octave}"


def parse_musicxml(path: pathlib.Path) -> ET.ElementTree:
    return ET.parse(path)


def extract_first_measures(tree: ET.ElementTree, limit: int = 20) -> List[LinearMeasure]:
    root = tree.getroot()
    # MusicXML namespace handling
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    part = root.find(f"{ns}part")
    measures = part.findall(f"{ns}measure") if part is not None else []
    linear: List[LinearMeasure] = []
    # Pre-fetch divisions (fallback 1 if not found)
    first_div_el = root.find(f".{ns}part/{ns}measure/{ns}attributes/{ns}divisions")
    default_divisions = int(first_div_el.text) if first_div_el is not None else 1
    for idx, m in enumerate(measures[:limit]):
        lin = LinearMeasure(linear_index=idx, original_measure=int(m.get('number','0')))
        current_time_staff = {1:0, 2:0}
        for note in m.findall(f"{ns}note"):
            is_rest = note.find(f"{ns}rest") is not None
            staff = int(note.findtext(f"{ns}staff", default='1'))
            dur_div = note.findtext(f"{ns}duration")
            grace = note.find(f"{ns}grace") is not None
            divisions_el = m.find(f"{ns}attributes/{ns}divisions")
            divisions = int(divisions_el.text) if divisions_el is not None else default_divisions
            sixteenth_unit = max(1, divisions // 4)  # avoid zero
            dur16 = 0 if grace or not dur_div else max(1, int(dur_div)//sixteenth_unit)
            notes_list: List[str] = []
            if not is_rest:
                step = note.findtext(f"{ns}pitch/{ns}step")
                alter = int(note.findtext(f"{ns}pitch/{ns}alter", default='0'))
                octave = int(note.findtext(f"{ns}pitch/{ns}octave"))
                notes_list.append(pitch_to_turkish(step, alter, octave))
            ev = NoteEvent(
                t=current_time_staff[staff],
                dur=dur16,
                notes=notes_list,
                grace=grace
            )
            target = lin.RH if staff == 1 else lin.LH
            target.append(ev)
            if not grace:
                current_time_staff[staff] += dur16
        linear.append(lin)
    return linear


def write_partial_outputs(measures: List[LinearMeasure], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Text
    txt_lines = [
        "Piano Sonata No.11 - Rondo alla Turca (PARTIAL 20 measures)",
        f"Generated UTC: {datetime.datetime.utcnow().isoformat()}",
        "Source: HAHA.musicxml",
        "---",
        "(This is an automatically generated partial preview. Full features pending.)"
    ]
    for m in measures:
        def hand(events: List[NoteEvent]) -> str:
            if not events:
                return "(rest)"
            tokens = []
            for e in events:
                if e.grace and e.notes:
                    tokens.append("{gr: " + " ".join(e.notes) + "}")
                    continue
                if not e.notes:
                    tokens.append(f"rest-{e.dur}")
                else:
                    note_token = "+".join(e.notes) + f"-{e.dur}"
                    tokens.append(note_token)
            return " ".join(tokens)
        line = f"L{m.linear_index}(orig m{m.original_measure}): RH: {hand(m.RH)} | LH: {hand(m.LH)}"
        txt_lines.append(line)
    (out_dir/"transcript_full.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    # JSON
    j = {
        "metadata": {
            "title": "Piano Sonata No.11 - Rondo alla Turca",
            "source_file": "HAHA.musicxml",
            "partial": True,
            "generated_utc": datetime.datetime.utcnow().isoformat(),
            "linear_measure_count": len(measures)
        },
        "measures": [
            {
                "linear_index": m.linear_index,
                "original_measure": m.original_measure,
                "RH": [e.__dict__ for e in m.RH],
                "LH": [e.__dict__ for e in m.LH],
            } for m in measures
        ]
    }
    (out_dir/"transcript_full.json").write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")

    # ABC (very minimal placeholder)
    abc_lines = [
        "X:1",
        "T:Rondo alla Turca (Partial 20 measures)",
        "M:2/4",
        "L:1/16",
        "Q:1/4=120",
        "K:C",
        "V:1 clef=treble",
        "V:2 clef=bass",
        "% Placeholder pitches (A for RH notes, C for LH notes, z for rests)"
    ]
    for m in measures:
        rh_bar = " ".join("z" if not e.notes else "".join('A' for _ in e.notes) for e in m.RH) or "z"
        lh_bar = " ".join("z" if not e.notes else "".join('C' for _ in e.notes) for e in m.LH) or "z"
        abc_lines.append(f"V:1 | {rh_bar} |")
        abc_lines.append(f"V:2 | {lh_bar} |")
    (out_dir/"transcript_full.abc").write_text("\n".join(abc_lines), encoding="utf-8")

    # measure_map.md (partial)
    md = [
        "# Measure Map (Partial)",
        "",
        "| LinearIndex | OriginalMeasure | Notes |",
        "|-------------|-----------------|-------|"
    ]
    for m in measures:
        md.append(f"| {m.linear_index} | {m.original_measure} | initial parse |")
    (out_dir/"measure_map.md").write_text("\n".join(md), encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_transcript.py HAHA.musicxml [LIMIT]", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tree = parse_musicxml(path)
    measures = extract_first_measures(tree, limit=limit)
    write_partial_outputs(measures, pathlib.Path('.'))
    print(f"Wrote partial outputs for {len(measures)} measures.")

if __name__ == "__main__":
    main()
