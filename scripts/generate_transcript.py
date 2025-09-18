#!/usr/bin/env python3
"""
MusicXML'den SADECE sağ el (RH, staff=1) notalarını çıkaran,
eşleştirilmiş repeat (|: :|) + volta (1.,2. ending) açılımını deterministik yapan
ve süre hesabını (TS + tempo) güvenli şekilde hesaplayan script.

Notlar:
- Repeat eşleştirme ön taramada yapılır; eşleşmeyen forward repeat'ler yok sayılır.
- Volta ending, aktif repeat pass numarasına göre atlanır.
- DC/DS/Coda/Fine uygulanmaz (uyarı verir).
- Ölçü süresi TS tabanlıdır; tempo <sound tempo> ve <metronome> desteklenir.

USAGE:
  python scripts/generate_transcript.py <input.musicxml> [LIMIT]
    LIMIT: all/-1 => tümü; pozitif sayı => ilk N ölçü (repeat açılımından sonra)

Outputs:
  transcript_rh.txt, transcript_rh.json, transcript_rh.lua, transcript_rh.abc, measure_map_rh.md
"""
import sys
import json
import datetime
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

__VERSION__ = "0.9.0"

DYNAMIC_TAGS = ["ppp","pp","p","mp","mf","f","ff","fff","fp","sfz"]
LETTER_TO_TURKISH = {"C":"Do","D":"Re","E":"Mi","F":"Fa","G":"Sol","A":"La","B":"Si"}
LETTER_TO_SEMITONE = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

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
    measure_beats: float = 0.0

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
    # 1) <sound tempo="...">
    sound = direction_el.find(ns.tag('sound')) if direction_el is not None else None
    if sound is not None and sound.get('tempo'):
        try:
            return float(sound.get('tempo'))
        except ValueError:
            pass
    # 2) <direction-type><metronome>
    dtyp = direction_el.find(ns.tag('direction-type')) if direction_el is not None else None
    if dtyp is not None:
        metro = dtyp.find(ns.tag('metronome'))
        if metro is not None:
            beat_unit = (metro.findtext(ns.tag('beat-unit')) or '').strip().lower()
            per_minute_txt = metro.findtext(ns.tag('per-minute'))
            try:
                per_minute = float(per_minute_txt) if per_minute_txt else None
            except ValueError:
                per_minute = None
            if per_minute:
                # beat-unit'i quarter BPM'e çevir
                unit_to_quarter = {
                    'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
                    'eighth': 0.5, '16th': 0.25, '32nd': 0.125, '64th': 0.0625
                }
                factor = unit_to_quarter.get(beat_unit, 1.0)
                return per_minute * factor
    return None

def collect_measures(root: ET.Element) -> List[ET.Element]:
    ns = NS(root)
    part = root.find(ns.tag('part'))
    if part is None:
        return []
    return list(part.findall(ns.tag('measure')))

def parse_endings(measures: List[ET.Element], ns: NS) -> Dict[int, Tuple[int, Set[int]]]:
    # ending_membership[i] = (end_idx, {1,2,...})
    ending_membership: Dict[int, Tuple[int, Set[int]]] = {}
    active_start = None
    active_nums: Set[int] = set()

    def parse_numbers(s: Optional[str]) -> Set[int]:
        if not s: return set()
        out: Set[int] = set()
        for p in s.replace(',', ' ').split():
            if p.isdigit():
                out.add(int(p))
        return out

    for idx, m in enumerate(measures):
        for barline in m.findall('.//' + ns.tag('barline')):
            ending = barline.find(ns.tag('ending'))
            if ending is None:
                continue
            etype = ending.get('type')
            nums = parse_numbers(ending.get('number'))
            if etype == 'start':
                active_start = idx
                active_nums = nums
            elif etype in ('stop', 'discontinue'):
                if active_start is not None:
                    for k in range(active_start, idx + 1):
                        ending_membership[k] = (idx, set(active_nums))
                active_start = None
                active_nums = set()
    return ending_membership

def pair_repeats(measures: List[ET.Element]) -> Tuple[Dict[int, Tuple[int,int]], Dict[int, Tuple[int,int]]]:
    """
    Repeat eşleştirme: forward (start_idx) -> (end_idx, times), backward (end_idx) -> (start_idx, times)
    Eşleşmeyen forward'lar yok sayılır.
    """
    if not measures:
        return {}, {}
    ns = NS(measures[0])
    stack: List[int] = []
    start_to_pair: Dict[int, Tuple[int,int]] = {}
    end_to_pair: Dict[int, Tuple[int,int]] = {}

    for i, m in enumerate(measures):
        # Bu ölçüdeki tüm barline'ları dolaş (ilgili direction'lar ayrı)
        for barline in m.findall('.//' + ns.tag('barline')):
            repeat = barline.find(ns.tag('repeat'))
            if repeat is None:
                continue
            direction = repeat.get('direction')
            if direction == 'forward':
                stack.append(i)
            elif direction == 'backward':
                if not stack:
                    continue  # eşleşecek forward yok, yok say
                start = stack.pop()
                times_attr = repeat.get('times')
                times = int(times_attr) if (times_attr and times_attr.isdigit()) else 2
                start_to_pair[start] = (i, times)
                end_to_pair[i] = (start, times)

    # Kalan forward'lar eşleşmedi, yok sayılır
    return start_to_pair, end_to_pair

def expand_playback(measures: List[ET.Element]) -> List[int]:
    """
    Eşleştirilmiş repeat + volta ile çalım sırası (ölçü indeksleri).
    """
    if not measures:
        return []
    ns = NS(measures[0])
    ending_membership = parse_endings(measures, ns)
    start_to_pair, end_to_pair = pair_repeats(measures)

    playback: List[int] = []
    i = 0
    # Stack: (start, end, pass_no, max_times)
    frames: List[Tuple[int,int,int,int]] = []

    raw_n = len(measures)
    max_steps = max(2000, raw_n * 10)  # güvenlik: orijinalin 10 katını aşma
    steps = 0

    while 0 <= i < raw_n and steps < max_steps:
        steps += 1

        # Volta ending: Aktif repeat varsa ve bu ölçü, geçerli pass için uygun değilse atla
        if frames:
            end_tup = ending_membership.get(i)
            if end_tup:
                end_idx, nums = end_tup
                start, end_, pass_no, max_times = frames[-1]
                if nums and (pass_no not in nums):
                    i = end_idx + 1
                    continue

        playback.append(i)

        # Eğer bu ölçü bir forward başlangıcıysa ve gerçek bir eşleşmesi varsa, iç içe repeat için çerçeve ekle
        if i in start_to_pair:
            end_idx, times = start_to_pair[i]
            frames.append((i, end_idx, 1, times))

        # Eğer bu ölçü bir backward bitişiyse ve üstteki çerçeve ile eşleşiyorsa
        if i in end_to_pair and frames:
            start_idx, times_b = end_to_pair[i]
            s, e, pass_no, max_times = frames[-1]
            if s == start_idx and e == i:
                # times uyumlaştır
                max_times = times_b if times_b else max_times
                if pass_no < max_times:
                    frames[-1] = (s, e, pass_no + 1, max_times)
                    i = s  # tekrar başına dön
                    continue
                else:
                    frames.pop()

        i += 1

    if steps >= max_steps:
        print(f"UYARI: Playback açılımı güvenlik sınırında durduruldu ({steps} adım). Repeat işaretlerini kontrol edin.", file=sys.stderr)

    return playback

def extract_measures(tree: ET.ElementTree, limit: Optional[int]=None) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    raw = collect_measures(root)
    order = expand_playback(raw)

    # LIMIT uygula
    if limit is not None and limit >= 0:
        order = order[:limit]

    linear: List[LinearMeasure] = []
    current_dynamic: Optional[str] = None
    current_tempo: float = 120.0
    current_divisions: int = 1

    global_time_beats: float = 0.0
    global_time_seconds: float = 0.0

    # Ölçü indexlerinden oluşan playback sırasını yürüt
    for lin_idx, meas_idx in enumerate(order):
        m = raw[meas_idx]
        no_txt = m.get('number', '0')
        try:
            meas_no = int(no_txt)
        except ValueError:
            meas_no = 0
        lm = LinearMeasure(linear_index=lin_idx, original_measure=meas_no)

        # Attributes
        attr = m.find(ns.tag('attributes'))
        div_el = attr.find(ns.tag('divisions')) if attr is not None else None
        if div_el is not None:
            try:
                current_divisions = max(1, int(div_el.text))
            except (TypeError, ValueError):
                pass
        divisions = max(current_divisions, 1)

        # Time signature -> ölçü süresi
        beats = None
        beat_type = None
        ts_el = attr.find(ns.tag('time')) if attr is not None else None
        if ts_el is not None:
            beats_txt = ts_el.findtext(ns.tag('beats'))
            beat_type_txt = ts_el.findtext(ns.tag('beat-type'))
            if beats_txt and beats_txt.isdigit() and beat_type_txt and beat_type_txt.isdigit():
                beats = int(beats_txt)
                beat_type = int(beat_type_txt)
        if beats and beat_type:
            measure_beats = beats * (4.0 / float(beat_type))
        else:
            # TS yoksa içerikten tahmin
            cur_div_pos = 0
            max_div_pos = 0
            for child in list(m):
                local = child.tag.split('}', 1)[-1] if '}' in child.tag else child.tag
                if local == 'backup':
                    dur_txt = child.findtext(ns.tag('duration'))
                    if dur_txt and dur_txt.isdigit():
                        cur_div_pos = max(0, cur_div_pos - int(dur_txt))
                elif local == 'forward':
                    dur_txt = child.findtext(ns.tag('duration'))
                    if dur_txt and dur_txt.isdigit():
                        cur_div_pos += int(dur_txt)
                        max_div_pos = max(max_div_pos, cur_div_pos)
                elif local == 'note':
                    dur_txt = child.findtext(ns.tag('duration'))
                    dur_div = int(dur_txt) if (dur_txt and dur_txt.isdigit()) else 0
                    # chord olmayan notalarda ilerle
                    if m.find(ns.tag('chord')) is None:
                        cur_div_pos += dur_div
                        max_div_pos = max(max_div_pos, cur_div_pos)
            measure_beats = max_div_pos / divisions if divisions else 0.0

        # Measure başı direction (tempo/dynamic)
        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic = dyn
                lm.new_dynamic = dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                # Clamp
                if tmp < 10.0 or tmp > 400.0:
                    print(f"UYARI: Olağan dışı tempo {tmp} BPM; clamp edildi.", file=sys.stderr)
                current_tempo = min(max(tmp, 10.0), 400.0)
                lm.tempo = current_tempo

        # RH olayları: ölçü içi akışı gez
        cur_div_pos = 0
        def is_chord(note_el: ET.Element) -> bool:
            return note_el.find(ns.tag('chord')) is not None

        for child in list(m):
            local = child.tag.split('}', 1)[-1] if '}' in child.tag else child.tag
            if local == 'direction':
                dyn = dynamic_from_direction(child, ns)
                if dyn:
                    current_dynamic = dyn
                    lm.new_dynamic = dyn
                tmp = tempo_from_direction(child, ns)
                if tmp:
                    if tmp < 10.0 or tmp > 400.0:
                        print(f"UYARI: Olağan dışı tempo {tmp} BPM (ölçü içi); clamp edildi.", file=sys.stderr)
                    current_tempo = min(max(tmp, 10.0), 400.0)
                    lm.tempo = current_tempo

            elif local == 'backup':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos = max(0, cur_div_pos - int(dur_txt))

            elif local == 'forward':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos += int(dur_txt)

            elif local == 'note':
                staff_txt = child.findtext(ns.tag('staff'), default='1')
                try:
                    staff = int(staff_txt)
                except ValueError:
                    staff = 1

                is_rest = child.find(ns.tag('rest')) is not None
                dur_txt = child.findtext(ns.tag('duration'))
                dur_div = int(dur_txt) if (dur_txt and dur_txt.isdigit()) else 0
                chord_flag = is_chord(child)
                grace = child.find(ns.tag('grace')) is not None

                step = child.findtext(ns.tag('pitch') + '/' + ns.tag('step')) if not is_rest else None
                alter = int(child.findtext(ns.tag('pitch') + '/' + ns.tag('alter'), default='0')) if step else 0
                octave = int(child.findtext(ns.tag('pitch') + '/' + ns.tag('octave'))) if step else 0

                if staff == 1:
                    ev_t_beats = global_time_beats + (cur_div_pos / max(1, divisions))
                    ev_d_beats = (dur_div / max(1, divisions))
                    pitches_tr: List[str] = []
                    pitches_sci: List[str] = []
                    midi: List[int] = []
                    if not is_rest and step is not None:
                        pitches_tr.append(turkish_name(step, alter, octave))
                        pitches_sci.append(sci_name(step, alter, octave))
                        midi.append(midi_number(step, alter, octave))

                    art: List[str] = []
                    if child.find('.//' + ns.tag('staccato')) is not None: art.append("staccato")
                    if child.find('.//' + ns.tag('accent'))   is not None: art.append("accent")
                    if child.find('.//' + ns.tag('tenuto'))   is not None: art.append("tenuto")
                    if child.find('.//' + ns.tag('strong-accent')) is not None: art.append("marcato")

                    slur_start = False; slur_end = False
                    for sl in child.findall('.//' + ns.tag('slur')):
                        t = sl.get('type')
                        if t == 'start': slur_start = True
                        elif t == 'stop': slur_end = True

                    lm.RH.append(NoteEvent(
                        t_beats=ev_t_beats, t_seconds=0.0,
                        dur_beats=ev_d_beats, dur_seconds=0.0,
                        pitches_tr=pitches_tr, pitches_sci=pitches_sci, midi=midi,
                        dyn=current_dynamic, art=art,
                        slur_start=slur_start, slur_end=slur_end, grace=grace
                    ))

                if not chord_flag:
                    cur_div_pos += dur_div

        # Zamanı saniyeye çevir ve global zamanda ilerle
        sec_per_beat = 60.0 / max(current_tempo, 1e-6)
        for ev in lm.RH:
            local_offset = ev.t_beats - global_time_beats
            ev.t_seconds = global_time_seconds + local_offset * sec_per_beat
            ev.dur_seconds = ev.dur_beats * sec_per_beat

        global_time_seconds += measure_beats * sec_per_beat
        global_time_beats += measure_beats
        lm.measure_beats = measure_beats

        linear.append(lm)

    # Bilgi
    print(f"[INFO] Raw measures: {len(raw)}", file=sys.stderr)
    print(f"[INFO] Expanded playback length: {len(order)}", file=sys.stderr)
    print(f"[INFO] Total duration (s): {global_time_seconds:.3f}", file=sys.stderr)
    if len(order) > len(raw) * 10:
        print(f"UYARI: Playback açılımı sıra dışı uzun: {len(raw)} -> {len(order)}. İşaretleri kontrol edin.", file=sys.stderr)

    return linear

def write_outputs(measures: List[LinearMeasure], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

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
        "MusicXML Sağ El Transcript (RH only, paired repeat+volta, TS-based)",
        f"Generated UTC: {now}",
        f"Linear measures: {len(measures)}",
        f"Events: {len(events)}",
        f"Script version: {__VERSION__}",
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
            'format_version': 9,
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
                 "| LinearIndex | OriginalMeasure | RH_events | MeasureBeats |",
                 "|------------:|---------------:|----------:|-------------:|"]
    for m in measures:
        map_lines.append(f"| {m.linear_index} | {m.original_measure} | {len(m.RH)} | {m.measure_beats:.3f} |")
    (out_dir/"measure_map_rh.md").write_text("\n".join(map_lines), encoding="utf-8")

    # Lua
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
        "-- Auto-generated RH transcription table (paired repeat+volta, TS-based)",
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

def parse_limit_arg(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    low = raw.strip().lower()
    if low in ('all', '-1'):
        return None
    try:
        v = int(low)
        return v if v >= 0 else None
    except ValueError:
        return None

def main():
    if len(sys.argv) < 2:
        print("USAGE: python scripts/generate_transcript.py <input.musicxml> [LIMIT]\n"
              "LIMIT = all/-1 (tümü) veya pozitif sayı (ilk N ölçü)\n"
              "Çıktılar: transcript_rh.txt, .json, .lua, .abc, .md\n", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"MusicXML not found: {path}", file=sys.stderr)
        sys.exit(2)

    limit = parse_limit_arg(sys.argv[2]) if len(sys.argv) > 2 else None

    tree = ET.parse(path)
    measures = extract_measures(tree, limit=limit)
    write_outputs(measures, pathlib.Path('.'))

if __name__ == '__main__':
    main()
