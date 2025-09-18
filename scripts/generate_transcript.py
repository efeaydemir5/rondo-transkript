#!/usr/bin/env python3
"""
Enhanced MusicXML -> multi-format transcript & Roblox Lua exporter.

Features:
- Extracts entire MusicXML piece, including all measures and basic repeats.
- Expands simple forward/backward repeats (not DC/segno/coda/ending).
- Outputs to .txt, .json, .abc, .md, .lua formats.
- LIMIT param: integer (number of measures to extract) or -1/'all' (extracts ALL).

USAGE:
    python scripts/generate_transcript.py <input.musicxml> [LIMIT]
If LIMIT is omitted, ALL measures are extracted (default).
"""

from __future__ import annotations
import sys, json, datetime, pathlib, xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

__VERSION__ = "0.3.5"

LETTER_TO_TURKISH = {"C":"Do","D":"Re","E":"Mi","F":"Fa","G":"Sol","A":"La","B":"Si"}
LETTER_TO_SEMITONE = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
DYNAMIC_TAGS = ["ppp","pp","p","mp","mf","f","ff","fff","fp","sfz"]

@dataclass
class Articulation:
    staccato: bool=False
    accent: bool=False
    tenuto: bool=False
    marcato: bool=False
    def tokens(self) -> List[str]:
        out=[]
        if self.staccato: out.append("staccato")
        if self.accent: out.append("accent")
        if self.tenuto: out.append("tenuto")
        if self.marcato: out.append("marcato")
        return out

@dataclass
class NoteEvent:
    t_beats: float
    t_seconds: float
    dur_beats: float
    dur_seconds: float
    pitches_tr: List[str]
    pitches_sci: List[str]
    midi: List[int]
    dyn: Optional[str]=None
    art: List[str]=field(default_factory=list)
    slur_start: bool=False
    slur_end: bool=False
    grace: bool=False
    hand: str="RH"

@dataclass
class LinearMeasure:
    linear_index: int
    original_measure: int
    RH: List[NoteEvent]=field(default_factory=list)
    LH: List[NoteEvent]=field(default_factory=list)
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

def parse_musicxml(path: pathlib.Path) -> ET.ElementTree:
    return ET.parse(path)

def collect_measures(root: ET.Element) -> List[ET.Element]:
    ns = NS(root)
    part = root.find(ns.tag('part'))
    if part is None:
        return []
    return list(part.findall(ns.tag('measure')))

def expand_repeats_basic(measures: List[ET.Element]) -> List[ET.Element]:
    if not measures:
        return []
    ns = NS(measures[0])
    forward_stack=[]
    expanded=[]
    used_backward=set()
    i=0
    while i < len(measures):
        m=measures[i]
        expanded.append(m)
        for barline in m.findall('.//'+ns.tag('barline')):
            repeat = barline.find(ns.tag('repeat'))
            if repeat is not None:
                direction = repeat.get('direction')
                if direction == 'forward':
                    forward_stack.append(i)
                elif direction == 'backward' and forward_stack:
                    if i not in used_backward:
                        start = forward_stack.pop()
                        seg = measures[start:i+1]
                        expanded.extend(seg)
                        used_backward.add(i)
        # UYARI: DC/segno/coda/ending şu anda desteklenmiyor!
        # Bu tür tekrarlar için log veya print eklenebilir.
        for sound in m.findall('.//'+ns.tag('sound')):
            if sound.get('dalsegno') or sound.get('dacapo') or sound.get('fine'):
                print(f"UYARI: Karmaşık tekrar (DC/segno/fine) algılandı; script sadece basit repeat açar!", file=sys.stderr)
        i+=1
    return expanded

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

def extract_measures(tree: ET.ElementTree, limit: int=-1) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    raw = collect_measures(root)
    expanded = expand_repeats_basic(raw)
    if limit not in (-1, None) and limit >= 0:
        expanded = expanded[:limit]

    linear: List[LinearMeasure] = []
    current_dynamic=None
    current_tempo=120.0
    staff_time_beats={1:0.0, 2:0.0}
    current_divisions = 1

    for lin_idx, m in enumerate(expanded):
        no_txt = m.get('number','0')
        try:
            meas_no = int(no_txt)
        except ValueError:
            meas_no = 0
        lm = LinearMeasure(linear_index=lin_idx, original_measure=meas_no)

        attr = m.find(ns.tag('attributes'))
        div_el = attr.find(ns.tag('divisions')) if attr is not None else None
        # DÜZELTME: divisions etiketi yoksa bir önceki divisions değeri korunur
        if div_el is not None and div_el.text and div_el.text.isdigit():
            current_divisions = int(div_el.text)
        divisions = current_divisions

        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic=dyn
                lm.new_dynamic=dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                current_tempo=tmp
                lm.tempo=tmp

        for note in m.findall(ns.tag('note')):
            is_rest = note.find(ns.tag('rest')) is not None
            staff_txt = note.findtext(ns.tag('staff'), default='1')
            try:
                staff = int(staff_txt)
            except ValueError:
                staff = 1
            duration_div = note.findtext(ns.tag('duration'))
            grace = note.find(ns.tag('grace')) is not None

            step = note.findtext(ns.tag('pitch')+'/'+ns.tag('step')) if not is_rest else None
            alter = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('alter'), default='0')) if step else 0
            octave = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('octave'))) if step else 0

            if grace or not duration_div or not duration_div.isdigit():
                dur_quarter=0.0
            else:
                dur_quarter=int(duration_div)/divisions
            dur_beats=dur_quarter

            start_beats=staff_time_beats.get(staff,0.0)
            start_seconds=(60.0/current_tempo)*start_beats

            pitches_tr=[]; pitches_sci=[]; midi=[]
            if not is_rest:
                pitches_tr.append(turkish_name(step, alter, octave))
                pitches_sci.append(sci_name(step, alter, octave))
                midi.append(midi_number(step, alter, octave))

            articulation = Articulation(
                staccato = note.find('.//'+ns.tag('staccato')) is not None,
                accent   = note.find('.//'+ns.tag('accent')) is not None,
                tenuto   = note.find('.//'+ns.tag('tenuto')) is not None,
                marcato  = note.find('.//'+ns.tag('strong-accent')) is not None,
            )
            slur_start=False; slur_end=False
            for sl in note.findall('.//'+ns.tag('slur')):
                t = sl.get('type')
                if t=='start': slur_start=True
                elif t=='stop': slur_end=True

            ev = NoteEvent(
                t_beats=start_beats,
                t_seconds=start_seconds,
                dur_beats=dur_beats,
                dur_seconds=(60.0/current_tempo)*dur_beats,
                pitches_tr=pitches_tr,
                pitches_sci=pitches_sci,
                midi=midi,
                dyn=current_dynamic,
                art=articulation.tokens(),
                slur_start=slur_start,
                slur_end=slur_end,
                grace=grace,
                hand='RH' if staff==1 else 'LH'
            )
            target = lm.RH if staff==1 else lm.LH
            target.append(ev)
            if not grace:
                staff_time_beats[staff]=start_beats+dur_beats
        linear.append(lm)

    def merge_ties(events: List[NoteEvent]) -> List[NoteEvent]:
        if not events: return []
        merged=[events[0]]
        for ev in events[1:]:
            last=merged[-1]
            if last.midi and ev.midi and last.midi == ev.midi and abs((last.t_beats+last.dur_beats)-ev.t_beats) < 1e-9:
                last.dur_beats += ev.dur_beats
                last.dur_seconds += ev.dur_seconds
                last.slur_end = last.slur_end or ev.slur_end
            else:
                merged.append(ev)
        return merged

    for lm in linear:
        lm.RH = merge_ties(lm.RH)
        lm.LH = merge_ties(lm.LH)

    return linear

def write_outputs(measures: List[LinearMeasure], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat()

    events=[]
    for m in measures:
        for ev in m.RH + m.LH:
            events.append({
                't': round(ev.t_seconds,6),
                'dur': round(ev.dur_seconds,6),
                'beats': round(ev.t_beats,6),
                'dur_beats': round(ev.dur_beats,6),
                'hand': ev.hand,
                'pitches_tr': ev.pitches_tr,
                'pitches_sci': ev.pitches_sci,
                'midi': ev.midi,
                'dyn': ev.dyn,
                'art': ev.art,
                'slur_start': ev.slur_start,
                'slur_end': ev.slur_end,
                'grace': ev.grace,
            })
    events.sort(key=lambda e: (e['t'], e['hand']))

    # TXT
    txt = [
        "Piano Sonata No.11 - Rondo alla Turca (Enhanced Export)",
        f"Generated UTC: {now}",
        "Source: HAHA.musicxml",
        f"Linear measures: {len(measures)}",
        f"Events: {len(events)}",
        "---",
        "time_sec\tdur_sec\thand\tpitches_sci\tdyn\tart"
    ]
    for e in events[:2000]:
        txt.append(f"{e['t']:.3f}\t{e['dur']:.3f}\t{e['hand']}\t{','.join(e['pitches_sci']) or 'rest'}\t{e['dyn'] or ''}\t{','.join(e['art'])}")
    (out_dir/"transcript_full.txt").write_text("\n".join(txt), encoding="utf-8")

    # JSON
    json_obj={
        'metadata':{
            'title':'Piano Sonata No.11 - Rondo alla Turca',
            'source_file':'HAHA.musicxml',
            'generated_utc':now,
            'measure_count':len(measures),
            'event_count':len(events),
            'format_version':3,
            'script_version':__VERSION__,
        },
        'events':events
    }
    (out_dir/"transcript_full.json").write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # ABC placeholder
    abc_lines=["X:1","T:Rondo alla Turca (Extracted)","M:2/4","L:1/16","Q:1/4=120","K:C","V:1 clef=treble","V:2 clef=bass"]
    (out_dir/"transcript_full.abc").write_text("\n".join(abc_lines), encoding="utf-8")

    # Measure map
    map_lines=["# Measure Map","",
               "| LinearIndex | OriginalMeasure | RH_events | LH_events |",
               "|------------:|---------------:|----------:|----------:|"]
    for m in measures:
        map_lines.append(f"| {m.linear_index} | {m.original_measure} | {len(m.RH)} | {len(m.LH)} |")
    (out_dir/"measure_map.md").write_text("\n".join(map_lines), encoding="utf-8")

    # Lua exporter helpers
    def lua_list(values):
        if not values:
            return "{}"
        formatted=[]
        for v in values:
            if isinstance(v,str):
                formatted.append("'" + v.replace("'", "\\'") + "'")
            else:
                formatted.append(str(v))
        return "{" + ",".join(formatted) + "}"

    lua_lines=[
        "-- Auto-generated transcription table",
        f"-- Generated UTC: {now}",
        f"-- Script version: {__VERSION__}",
        "return {",
        "  metadata = {",
        "    title = 'Piano Sonata No.11 - Rondo alla Turca',",
        f"    generated_utc = '{now}',",
        f"    event_count = {len(events)},",
        f"    script_version = '{__VERSION__}',",
        "  },",
        "  notes = {"
    ]
    for e in events:
        dyn_field = f"'{e['dyn']}'" if e['dyn'] else "nil"
        lua_lines.append(
            "    {t=%.6f, dur=%.6f, hand='%s', pitches_sci=%s, pitches_tr=%s, midi=%s, dyn=%s, art=%s, slur_start=%s, slur_end=%s, grace=%s}," % (
                e['t'],
                e['dur'],
                e['hand'],
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
    (out_dir/"transcript_full.lua").write_text("\n".join(lua_lines), encoding="utf-8")

    print(f"Wrote {len(events)} events across {len(measures)} measures.")

def main():
    if len(sys.argv) < 2:
        print("USAGE: python scripts/generate_transcript.py <input.musicxml> [LIMIT]\n"
              "LIMIT = number of measures, or -1/all (default: all measures, including repeats)\n"
              "Example: python scripts/generate_transcript.py mypiece.musicxml 12", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"MusicXML not found: {path}", file=sys.stderr)
        sys.exit(2)
    if len(sys.argv) > 2:
        raw_limit = sys.argv[2]
        limit = -1 if raw_limit.lower() in ('-1','all') else int(raw_limit)
    else:
        limit = -1

    tree = parse_musicxml(path)
    measures = extract_measures(tree, limit=limit)
    write_outputs(measures, pathlib.Path('.'))

if __name__ == '__main__':
    main()
