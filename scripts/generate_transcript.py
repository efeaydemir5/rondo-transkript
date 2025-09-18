#!/usr/bin/env python3
"""
MusicXML'den SADECE sağ el (RH, staff=1) notalarını çıkaran, repeat açmayan ve loop hatası yapmayan transcript scripti.
Tüm ölçüleri orijinal sırayla bir kez işler, repeat açmaz.
Çıktılar: .txt, .json, .lua, .abc, .md (sadece RH)
USAGE:
    python scripts/generate_transcript.py <input.musicxml>
"""

import sys
import json
import datetime
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

__VERSION__ = "0.4.0"

LETTER_TO_TURKISH = {"C": "Do", "D": "Re", "E": "Mi", "F": "Fa", "G": "Sol", "A": "La", "B": "Si"}
LETTER_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DYNAMIC_TAGS = ["ppp","pp","p","mp","mf","f","ff","fff","fp","sfz"]

@dataclass
class NoteEvent:
    t_beats: float
    t_seconds: float
    dur_beats: float
    dur_seconds: float
    pitches_tr: List[str]
    pitches_sci: List[str]
    midi: List[int]
    dyn: Optional[str] = None
    art: List[str] = field(default_factory=list)
    slur_start: bool = False
    slur_end: bool = False
    grace: bool = False

@dataclass
class LinearMeasure:
    linear_index: int
    original_measure: int
    RH: List[NoteEvent] = field(default_factory=list)
    tempo: Optional[float]=None
    new_dynamic: Optional[str]=None

class NS:
    def __init__(self, root: ET.Element):
        if root.tag.startswith('{'):
            self.ns = root.tag.split('}')[0].strip('{')
        else:
            self.ns = None
    def tag(self, local: str) -> str:
        return f"{{{self.ns}}}{local}" if self.ns else local

def midi_number(step: str, alter: int, octave: int) -> int:
    return (octave + 1)*12 + LETTER_TO_SEMITONE[step] + alter

def sci_name(step: str, alter: int, octave: int) -> str:
    acc = {1:'#', -1:'b'}.get(alter,'')
    return f"{step}{acc}{octave}"

def turkish_name(step: str, alter: int, octave: int) -> str:
    base = LETTER_TO_TURKISH[step]
    if alter == 1: base += '#'
    elif alter == -1: base += 'b'
    return f"{base}{octave}"

def dynamic_from_direction(direction_el: ET.Element, ns: NS) -> Optional[str]:
    dtyp = direction_el.find(ns.tag('direction-type')) if direction_el is not None else None
    if dtyp is None: return None
    for dtag in DYNAMIC_TAGS:
        if dtyp.find(ns.tag(dtag)) is not None:
            return dtag
    return None

def tempo_from_direction(direction_el: ET.Element, ns: NS) -> Optional[float]:
    sound = direction_el.find(ns.tag('sound')) if direction_el is not None else None
    if sound is not None and sound.get('tempo'):
        try:
            return float(sound.get('tempo'))
        except ValueError:
            return None
    return None

def extract_measures(tree: ET.ElementTree) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    # Sadece repeat açmadan, orijinal ölçüleri sırayla al
    part = root.find(ns.tag('part'))
    measures = list(part.findall(ns.tag('measure'))) if part is not None else []

    linear: List[LinearMeasure] = []
    current_dynamic = None
    current_tempo = 120.0
    current_divisions = 1
    staff_time_beats = 0.0  # Sadece RH için

    for lin_idx, m in enumerate(measures):
        no_txt = m.get('number', '0')
        try:
            meas_no = int(no_txt)
        except ValueError:
            meas_no = 0
        lm = LinearMeasure(linear_index=lin_idx, original_measure=meas_no)

        attr = m.find(ns.tag('attributes'))
        div_el = attr.find(ns.tag('divisions')) if attr is not None else None
        if div_el is not None and div_el.text and div_el.text.isdigit():
            current_divisions = int(div_el.text)
        divisions = current_divisions

        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic = dyn
                lm.new_dynamic = dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                current_tempo = tmp
                lm.tempo = tmp

        for note in m.findall(ns.tag('note')):
            # Sadece staff=1 (RH) için
            staff_txt = note.findtext(ns.tag('staff'), default='1')
            try:
                staff = int(staff_txt)
            except ValueError:
                staff = 1
            if staff != 1:
                continue  # Sadece RH çıkar

            is_rest = note.find(ns.tag('rest')) is not None
            duration_div = note.findtext(ns.tag('duration'))
            grace = note.find(ns.tag('grace')) is not None

            step = note.findtext(ns.tag('pitch')+'/'+ns.tag('step')) if not is_rest else None
            alter = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('alter'), default='0')) if step else 0
            octave = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('octave'))) if step else 0

            if grace or not duration_div or not duration_div.isdigit():
                dur_quarter = 0.0
            else:
                dur_quarter = int(duration_div)/divisions
            dur_beats = dur_quarter

            start_beats = staff_time_beats
            start_seconds = (60.0/current_tempo)*start_beats

            pitches_tr = []; pitches_sci = []; midi = []
            if not is_rest:
                pitches_tr.append(turkish_name(step, alter, octave))
                pitches_sci.append(sci_name(step, alter, octave))
                midi.append(midi_number(step, alter, octave))

            articulation = []
            if note.find('.//'+ns.tag('staccato')) is not None:
                articulation.append("staccato")
            if note.find('.//'+ns.tag('accent')) is not None:
                articulation.append("accent")
            if note.find('.//'+ns.tag('tenuto')) is not None:
                articulation.append("tenuto")
            if note.find('.//'+ns.tag('strong-accent')) is not None:
                articulation.append("marcato")
            slur_start = False; slur_end = False
            for sl in note.findall('.//'+ns.tag('slur')):
                t = sl.get('type')
                if t == 'start': slur_start = True
                elif t == 'stop': slur_end = True

            ev = NoteEvent(
                t_beats=start_beats,
                t_seconds=start_seconds,
                dur_beats=dur_beats,
                dur_seconds=(60.0/current_tempo)*dur_beats,
                pitches_tr=pitches_tr,
                pitches_sci=pitches_sci,
                midi=midi,
                dyn=current_dynamic,
                art=articulation,
                slur_start=slur_start,
                slur_end=slur_end,
                grace=grace,
            )
            lm.RH.append(ev)
            if not grace:
                staff_time_beats = start_beats + dur_beats
        linear.append(lm)
    return linear

def write_outputs(measures: List[LinearMeasure], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat()

    events = []
    for m in measures:
        for ev in m.RH:
            events.append({
                't': round(ev.t_seconds, 6),
                'dur': round(ev.dur_seconds, 6),
                'beats': round(ev.t_beats, 6),
                'dur_beats': round(ev.dur_beats, 6),
                'pitches_tr': ev.pitches_tr,
                'pitches_sci': ev.pitches_sci,
                'midi': ev.midi,
                'dyn': ev.dyn,
                'art': ev.art,
                'slur_start': ev.slur_start,
                'slur_end': ev.slur_end,
                'grace': ev.grace,
            })
    events.sort(key=lambda e: (e['t']))

    # TXT
    txt = [
        "MusicXML Sağ El Transcript (RH only)",
        f"Generated UTC: {now}",
        f"Linear measures: {len(measures)}",
        f"Events: {len(events)}",
        "---",
        "time_sec\tdur_sec\tpitches_sci\tdyn\tart"
    ]
    for e in events:
        txt.append(f"{e['t']:.3f}\t{e['dur']:.3f}\t{','.join(e['pitches_sci']) or 'rest'}\t{e['dyn'] or ''}\t{','.join(e['art'])}")
    (out_dir/"transcript_rh.txt").write_text("\n".join(txt), encoding="utf-8")

    # JSON
    json_obj = {
        'metadata': {
            'generated_utc': now,
            'measure_count': len(measures),
            'event_count': len(events),
            'format_version': 4,
            'script_version': __VERSION__,
        },
        'events': events
    }
    (out_dir/"transcript_rh.json").write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # ABC placeholder
    abc_lines = ["X:1", "T:Extracted RH", "M:4/4", "L:1/16", "Q:1/4=120", "K:C", "V:1 clef=treble"]
    (out_dir/"transcript_rh.abc").write_text("\n".join(abc_lines), encoding="utf-8")

    # Measure map
    map_lines = ["# Measure Map", "",
               "| LinearIndex | OriginalMeasure | RH_events |",
               "|------------:|---------------:|----------:|"]
    for m in measures:
        map_lines.append(f"| {m.linear_index} | {m.original_measure} | {len(m.RH)} |")
    (out_dir/"measure_map_rh.md").write_text("\n".join(map_lines), encoding="utf-8")

    # Lua exporter helpers
    def lua_list(values):
        if not values:
            return "{}"
        formatted = []
        for v in values:
            if isinstance(v, str):
                formatted.append("'" + v.replace("'", "\\'") + "'")
            else:
                formatted.append(str(v))
        return "{" + ",".join(formatted) + "}"

    lua_lines = [
        "-- Auto-generated RH transcription table",
        f"-- Generated UTC: {now}",
        f"-- Script version: {__VERSION__}",
        "return {",
        "  metadata = {",
        "    generated_utc = '" + now + "',",
        f"    event_count = {len(events)},",
        f"    script_version = '{__VERSION__}',",
        "  },",
        "  notes = {"
    ]
    for e in events:
        dyn_field = f"'{e['dyn']}'" if e['dyn'] else "nil"
        lua_lines.append(
            "    {t=%.6f, dur=%.6f, pitches_sci=%s, pitches_tr=%s, midi=%s, dyn=%s, art=%s, slur_start=%s, slur_end=%s, grace=%s}," % (
                e['t'],
                e['dur'],
                lua_list(e['pitches_sci']),
                lua_list(e['pitches_tr']),
                lua_list(e['midi']),
                dyn_field,
                lua_list(e['art']),
                'true' if e['slur_start'] else 'false',
                'true' if e['slur_end'] else 'false',
                'true' if e['grace'] else 'false'
            )
        )
    lua_lines.append("  }")
    lua_lines.append("}")
    (out_dir/"transcript_rh.lua").write_text("\n".join(lua_lines), encoding="utf-8")

    print(f"Wrote {len(events)} RH events across {len(measures)} measures.")

def main():
    if len(sys.argv) < 2:
        print("USAGE: python scripts/generate_transcript.py <input.musicxml>\n"
              "Sadece sağ el (RH) çıkarılır. Repeat açılmaz, karmaşık tekrarlar işlenmez.\n"
              "Çıktılar: transcript_rh.txt, .json, .lua, .abc, .md\n", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"MusicXML not found: {path}", file=sys.stderr)
        sys.exit(2)

    tree = ET.parse(path)
    measures = extract_measures(tree)
    write_outputs(measures, pathlib.Path('.'))

if __name__ == '__main__':
    main()
