#!/usr/bin/env python3
"""Enhanced MusicXML -> transcript & Lua exporter.

Features added:
 - Full piece extraction (use limit=-1 or omit for all measures)
 - Tie merging (durations of tied notes combined)
 - Dynamics parsing (p, mp, mf, f, ff, etc.)
 - Articulations (staccato, accent, tenuto, marcato)
 - Slur start / end flags
 - Tempo changes (basic support via <sound tempo=''>) applied at note onset
 - Outputs:
     * transcript_full.txt  (human-readable summary)
     * transcript_full.json (structured events)
     * transcript_full.abc  (very rough placeholder)
     * measure_map.md       (measure mapping table)
     * transcript_full.lua  (Lua table ready for Roblox scripting)
 - Turkish pitch names retained, plus MIDI number & scientific pitch

Limitations / TODO:
 - Repeat & ending expansion is still basic: handles simple forward/backward repeats once, ignores complex nested endings.
 - Grace notes: kept with dur=0 (not inserted into timing advance).
 - Multiple voices on same staff merged in insertion order.

Usage:
    python scripts/generate_transcript.py HAHA.musicxml [LIMIT]
Where LIMIT is integer number of linear measures after (optional) repeat expansion, or -1 / 'all' for full piece.

Roblox Lua usage idea:
    local score = require(path_to_module).notes
    -- Each entry: {t=seconds, dur=seconds, hand='RH'|'LH', pitches={'C4',...}, midi={60,...}, dyn='mf', art={...}}

"""
from __future__ import annotations
import sys, json, datetime, pathlib, xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# --- Pitch / Music helpers --------------------------------------------------
LETTER_TO_TURKISH = {"C":"Do","D":"Re","E":"Mi","F":"Fa","G":"Sol","A":"La","B":"Si"}
LETTER_TO_SEMITONE = {"C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11}
DYNAMIC_TAGS = ["ppp","pp","p","mp","mf","f","ff","fff","fp","sfz"]

dataclass
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

dataclass
class NoteEvent:
    t_beats: float
    t_seconds: float
    dur_beats: float
    dur_seconds: float
    pitches: List[str]             # Turkish names (Do4, etc.)
    sci: List[str]                 # Scientific pitch (C#4)
    midi: List[int]
    dyn: Optional[str]=None
    art: List[str]=field(default_factory=list)
    slur_start: bool=False
    slur_end: bool=False
    grace: bool=False
    hand: str="RH"

dataclass
class LinearMeasure:
    linear_index: int
    original_measure: int
    RH: List[NoteEvent]=field(default_factory=list)
    LH: List[NoteEvent]=field(default_factory=list)
    tempo: Optional[float]=None
    new_dynamic: Optional[str]=None

# ---------------------------------------------------------------------------

def midi_number(step: str, alter: int, octave: int) -> int:
    return (octave + 1)*12 + LETTER_TO_SEMITONE[step] + alter

def sci_name(step: str, alter: int, octave: int) -> str:
    accidental = {1:'♯', -1:'♭'}.get(alter,'')
    # Use #/b ASCII for Roblox compatibility
    accidental_ascii = {1:'#', -1:'b'}.get(alter,'')
    return f"{step}{accidental_ascii}{octave}"

def turkish_name(step: str, alter: int, octave: int) -> str:
    base = LETTER_TO_TURKISH[step]
    if alter == 1:
        base += '#'
    elif alter == -1:
        base += 'b'
    return f"{base}{octave}"

# Namespace helper
class NS:
    def __init__(self, root: ET.Element):
        if root.tag.startswith('{'):
            self.ns = root.tag.split('}')[0].strip('{')
        else:
            self.ns = None
    def tag(self, local: str) -> str:
        return f"{{{self.ns}}}{local}" if self.ns else local

# Repeat handling (very basic)
dataclass
class RepeatInfo:
    forward_index: int
    backward_index: int

# ---------------------------------------------------------------------------

def parse_musicxml(path: pathlib.Path) -> ET.ElementTree:
    return ET.parse(path)


def collect_measures(root: ET.Element) -> List[ET.Element]:
    ns = NS(root)
    part = root.find(ns.tag('part'))
    if part is None:
        return []
    return list(part.findall(ns.tag('measure')))


def expand_repeats_basic(measures: List[ET.Element]) -> List[ET.Element]:
    # Handles single level forward/backward repeat marks (direction="forward"/"backward") once.
    ns = NS(measures[0]) if measures else None
    forward_stack = []
    expanded = []
    i=0
    used_backward=set()
    while i < len(measures):
        m = measures[i]
        expanded.append(m)
        # Detect repeat signs
        for barline in m.findall('.//'+(ns.tag('barline') if ns else 'barline')):
            repeat = barline.find((ns.tag('repeat') if ns else 'repeat'))
            if repeat is not None:
                direction = repeat.get('direction')
                if direction == 'forward':
                    forward_stack.append(i)
                elif direction == 'backward' and forward_stack:
                    if i not in used_backward:  # do only once
                        start = forward_stack.pop()
                        segment = measures[start:i+1]
                        expanded.extend(segment)  # second pass
                        used_backward.add(i)
        i += 1
    return expanded


def dynamic_from_direction(direction_el: ET.Element, ns: NS) -> Optional[str]:
    dyns = direction_el.find(ns.tag('direction-type')) if direction_el is not None else None
    if dyns is None:
        return None
    for dtag in DYNAMIC_TAGS:
        if dyns.find(ns.tag(dtag)) is not None:
            return dtag
    return None


def tempo_from_direction(direction_el: ET.Element, ns: NS) -> Optional[float]:
    # <sound tempo="120"/>
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
    raw_measures = collect_measures(root)
    expanded = expand_repeats_basic(raw_measures)
    if limit not in (-1, None) and limit >= 0:
        expanded = expanded[:limit]
    linear: List[LinearMeasure] = []

    current_dynamic = None
    current_tempo = 120.0

    # For timing we track per staff accumulated beats.
    staff_time_beats = {1:0.0, 2:0.0}

    for lin_idx, m in enumerate(expanded):
        meas_no = int(m.get('number','0'))
        lm = LinearMeasure(linear_index=lin_idx, original_measure=meas_no)
        # divisions for this measure
        div_el = m.find(ns.tag('attributes')+'/'+ns.tag('divisions')) if m.find(ns.tag('attributes')) is not None else None
        divisions = int(div_el.text) if div_el is not None else 1

        # Directions (dynamics / tempo / slurs?)
        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic = dyn
                lm.new_dynamic = dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                current_tempo = tmp
                lm.tempo = tmp

        # Process notes
        for note in m.findall(ns.tag('note')):
            is_rest = note.find(ns.tag('rest')) is not None
            staff = int(note.findtext(ns.tag('staff'), default='1'))
            voice = note.findtext(ns.tag('voice'))  # unused currently
            duration_div = note.findtext(ns.tag('duration'))
            grace = note.find(ns.tag('grace')) is not None

            step = note.findtext(ns.tag('pitch')+'/'+ns.tag('step')) if not is_rest else None
            alter = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('alter'), default='0')) if step else 0
            octave = int(note.findtext(ns.tag('pitch')+'/'+ns.tag('octave'))) if step else 0

            # Convert duration: duration_divisions / divisions = quarter notes
            if grace or not duration_div:
                dur_quarter = 0.0
            else:
                dur_quarter = int(duration_div)/divisions
            dur_beats = dur_quarter  # treat quarter note = 1 beat

            # Start times (grace notes share time with following => do not advance before)
            start_beats = staff_time_beats[staff]
            start_seconds = (60.0/current_tempo) * start_beats

            # Build pitch lists
            turkish_list=[]; sci_list=[]; midi_list=[]
            if not is_rest:
                turkish_list.append(turkish_name(step, alter, octave))
                sci_list.append(sci_name(step, alter, octave))
                midi_list.append(midi_number(step, alter, octave))

            # Articulations
            articulation = Articulation(
                staccato = note.find('.//'+ns.tag('staccato')) is not None,
                accent   = note.find('.//'+ns.tag('accent')) is not None,
                tenuto   = note.find('.//'+ns.tag('tenuto')) is not None,
                marcato  = note.find('.//'+ns.tag('strong-accent')) is not None,
            )

            # Slurs
            slur_start = False; slur_end = False
            for sl in note.findall('.//'+ns.tag('slur')):
                stype = sl.get('type')
                if stype == 'start': slur_start = True
                elif stype == 'stop': slur_end = True

            ev = NoteEvent(
                t_beats=start_beats,
                t_seconds=start_seconds,
                dur_beats=dur_beats,
                dur_seconds=(60.0/current_tempo)*dur_beats,
                pitches=turkish_list,
                sci=sci_list,
                midi=midi_list,
                dyn=current_dynamic,
                art=articulation.tokens(),
                slur_start=slur_start,
                slur_end=slur_end,
                grace=grace,
                hand='RH' if staff == 1 else 'LH'
            )
            target = lm.RH if staff == 1 else lm.LH
            target.append(ev)

            if not grace:
                staff_time_beats[staff] += dur_beats
        linear.append(lm)

    # Tie merging (simple): merge consecutive notes with same first midi in same hand & beat continuity with zero gap.
    def merge_ties(events: List[NoteEvent]) -> List[NoteEvent]:
        if not events: return []
        merged=[events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if last.midi and ev.midi and last.midi==ev.midi and abs((last.t_beats+last.dur_beats)-ev.t_beats) < 1e-9:
                # merge
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

# ---------------------------------------------------------------------------

def write_outputs(measures: List[LinearMeasure], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat()

    # Flatten events for JSON & Lua
    all_events: List[Dict] = []
    for m in measures:
        for ev in m.RH + m.LH:
            all_events.append({
                't': round(ev.t_seconds,6),
                'dur': round(ev.dur_seconds,6),
                'beats': round(ev.t_beats,6),
                'dur_beats': round(ev.dur_beats,6),
                'hand': ev.hand,
                'pitches_tr': ev.pitches,
                'pitches_sci': ev.sci,
                'midi': ev.midi,
                'dyn': ev.dyn,
                'art': ev.art,
                'slur_start': ev.slur_start,
                'slur_end': ev.slur_end,
                'grace': ev.grace,
            })
    # Sort by absolute time then hand
    all_events.sort(key=lambda e: (e['t'], e['hand']))

    # TXT
    txt_lines = [
        "Piano Sonata No.11 - Rondo alla Turca (FULL / Enhanced Export)",
        f"Generated UTC: {now}",
        "Source: HAHA.musicxml",
        f"Linear measures: {len(measures)}",
        f"Events: {len(all_events)}",
        "---",
        "Columns: time_sec | dur_sec | hand | pitches_sci | dyn | art"
    ]
    for e in all_events[:1000]:  # safeguard if extremely large
        txt_lines.append(f"{e['t']:.3f}\t{e['dur']:.3f}\t{e['hand']}\t{','.join(e['pitches_sci']) or 'rest'}\t{e['dyn'] or ''}\t{','.join(e['art'])}")
    (out_dir/"transcript_full.txt").write_text("\n".join(txt_lines), encoding='utf-8')

    # JSON
    json_obj = {
        'metadata': {
            'title': 'Piano Sonata No.11 - Rondo alla Turca',
            'source_file': 'HAHA.musicxml',
            'generated_utc': now,
            'measure_count': len(measures),
            'event_count': len(all_events),
            'format_version': 2
        },
        'events': all_events
    }
    (out_dir/"transcript_full.json").write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding='utf-8')

    # Minimal ABC placeholder (unchanged idea)
    abc_lines = ["X:1","T:Rondo alla Turca (Extracted)","M:2/4","L:1/16","Q:1/4=120","K:C","V:1 clef=treble","V:2 clef=bass"]
    (out_dir/"transcript_full.abc").write_text("\n".join(abc_lines), encoding='utf-8')

    # Measure map
    map_lines = ["# Measure Map","","| LinearIndex | OriginalMeasure | RH_events | LH_events |","|------------:|---------------:|----------:|----------:|"]
    for m in measures:
        map_lines.append(f"| {m.linear_index} | {m.original_measure} | {len(m.RH)} | {len(m.LH)} |")
    (out_dir/"measure_map.md").write_text("\n".join(map_lines), encoding='utf-8')

    # Lua export
    lua_lines = ["-- Auto-generated transcription table","return {","  metadata = {",f"    title = 'Piano Sonata No.11 - Rondo alla Turca',",f"    generated_utc = '{now}',",f"    event_count = {len(all_events)},", "  },","  notes = {"
    for e in all_events:
        lua_lines.append(
            "    {t="+f"{e['t']:.6f}, dur={e['dur']:.6f}, hand='{e['hand']}', pitches={{{}}}, midi={{{}}}, dyn='{dyn}', art={{{}}}, grace={{g}}".format(
                ','.join(e['pitches_sci']),
                ','.join(str(m) for m in e['midi']),
                dyn=(e['dyn'] or ''),
                g='true' if e['grace'] else 'false',
                art=','.join("'"+a+"'" for a in e['art'])
            ).replace('dyn=''','dyn=nil')
        )
    lua_lines.append("  }","}")
    (out_dir/"transcript_full.lua").write_text("\n".join(lua_lines), encoding='utf-8')

    print(f"Wrote {len(all_events)} events across {len(measures)} measures.")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_transcript.py HAHA.musicxml [LIMIT]", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"MusicXML not found: {path}", file=sys.stderr)
        sys.exit(2)
    if len(sys.argv) > 2:
        raw_limit = sys.argv[2]
        limit = -1 if raw_limit in ('-1','all','ALL') else int(raw_limit)
    else:
        limit = -1
    tree = parse_musicxml(path)
    measures = extract_measures(tree, limit=limit)
    write_outputs(measures, pathlib.Path('.'))

if __name__ == '__main__':
    main()