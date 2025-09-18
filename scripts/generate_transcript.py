#!/usr/bin/env python3
"""
MusicXML'den SADECE sağ el (RH, staff=1) notalarını çıkaran,
repeat (ileri/geri) ve volta (1.,2. son) işaretlerini açan ve
ölçü süresini time signature + divisions + tempo ile doğru hesaplayan script.

- Müziğin gerçek çalınış sırasını bozmadan tekrarları genişletir (|: ... :| ve 1.,2. ending).
- RH dışındaki içerik zaman hesabı için dikkate alınır (ölçü süresi doğru ilerler),
  ama çıktıya sadece RH olayları yazılır.
- Tempo: <sound tempo="..."> ve/veya <direction-type><metronome> desteklenir.
- Ölçü süresi: Önce time signature'dan hesaplanır; içerik analizi ile bulunan süre bunu aşıyorsa clamp edilir.
  Pickup (eksik) ölçüde içerik süresi kullanılır.

USAGE:
    python scripts/generate_transcript.py <input.musicxml> [LIMIT]
      LIMIT: all/-1 => tümü, yoksa/pozitif => ilk N ölçü (repeat genişletildikten sonra)

Outputs:
    transcript_full.txt, transcript_full.json, transcript_full.lua, transcript_full.abc, measure_map.md
"""

import sys
import json
import datetime
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

__VERSION__ = "0.7.0"

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
    # 2) <direction-type><metronome><beat-unit>...</beat-unit><per-minute>...</per-minute>
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
                # beat-unit'i çeyreğe çevir
                unit_map = {
                    'whole': 1/4.0, 'half': 1/2.0, 'quarter': 1.0,
                    'eighth': 2.0, '16th': 4.0, '32nd': 8.0, '64th': 16.0
                }
                factor = unit_map.get(beat_unit, 1.0)  # bilinmiyorsa çeyrek varsay
                return per_minute * factor
    return None

def collect_measures(root: ET.Element) -> List[ET.Element]:
    ns = NS(root)
    part = root.find(ns.tag('part'))
    if part is None:
        return []
    return list(part.findall(ns.tag('measure')))

def parse_ending_ranges(measures: List[ET.Element], ns: NS) -> Dict[int, Tuple[int, Set[int]]]:
    ending_membership: Dict[int, Tuple[int, Set[int]]] = {}
    active_start = None
    active_numbers: Set[int] = set()

    def parse_numbers(text: Optional[str]) -> Set[int]:
        if not text:
            return set()
        nums = set()
        for part in text.replace(',', ' ').split():
            if part.isdigit():
                nums.add(int(part))
        return nums

    for idx, m in enumerate(measures):
        for barline in m.findall('.//' + ns.tag('barline')):
            ending = barline.find(ns.tag('ending'))
            if ending is None:
                continue
            etype = ending.get('type')  # start, stop, discontinue
            numbers = parse_numbers(ending.get('number'))
            if etype == 'start':
                active_start = idx
                active_numbers = numbers if numbers else set()
            elif etype in ('stop', 'discontinue'):
                if active_start is not None:
                    for k in range(active_start, idx + 1):
                        ending_membership[k] = (idx, set(active_numbers))
                active_start = None
                active_numbers = set()
    return ending_membership

def expand_repeats_with_volta(measures: List[ET.Element]) -> List[ET.Element]:
    if not measures:
        return []
    ns = NS(measures[0])
    ending_membership = parse_ending_ranges(measures, ns)

    expanded: List[ET.Element] = []
    i = 0
    frames: List[Tuple[int, Optional[int], int, int]] = []  # (start_idx, end_idx, pass_no, max_times)

    safety_steps = 0
    max_steps = 200000  # güvenlik

    while i < len(measures):
        safety_steps += 1
        if safety_steps > max_steps:
            print("UYARI: Repeat açılımında güvenlik sınırı aşıldı, olası döngü. Açılım durduruldu.", file=sys.stderr)
            break

        m = measures[i]

        # Karmaşık işaretler uyarısı (uygulamıyoruz)
        for direction in m.findall(ns.tag('direction')):
            sound = direction.find(ns.tag('sound'))
            if sound is not None and (sound.get('dacapo') or sound.get('dalsegno') or sound.get('fine') or sound.get('tocoda') or sound.get('coda') or sound.get('segno')):
                print(f"UYARI: DC/DS/Coda/Fine algılandı (ölçü {m.get('number','?')}); bu sürüm sadece ileri/geri repeat ve ending açar.", file=sys.stderr)

        # Volta uygun değilse atla
        if frames:
            end_tup = ending_membership.get(i)
            if end_tup:
                end_idx, ending_nums = end_tup
                start_idx, end_idx_frame, pass_no, max_times = frames[-1]
                if ending_nums and (pass_no not in ending_nums):
                    i = end_idx + 1
                    continue

        expanded.append(m)

        forward_here = False
        backward_here = False
        backward_times = None

        for barline in m.findall('.//' + ns.tag('barline')):
            repeat = barline.find(ns.tag('repeat'))
            if repeat is not None:
                direction = repeat.get('direction')
                if direction == 'forward':
                    forward_here = True
                elif direction == 'backward':
                    backward_here = True
                    t = repeat.get('times')
                    if t and t.isdigit():
                        backward_times = int(t)

        if forward_here:
            frames.append((i, None, 1, backward_times or 2))  # şimdilik varsayılan 2

        if backward_here:
            if frames:
                start_idx, _, pass_no, max_times = frames[-1]
                if backward_times is not None:
                    max_times = backward_times
                if pass_no < max_times:
                    frames[-1] = (start_idx, i, pass_no + 1, max_times)
                    i = start_idx
                    continue
                else:
                    frames.pop()

        i += 1

    # Aşırı büyümeyi tespit et ve uyar
    if len(expanded) > max(1, len(measures)) * 20:
        print(f"UYARI: Repeat açılımı sıra dışı büyük: {len(measures)} -> {len(expanded)} ölçü. İşaretleri kontrol edin.", file=sys.stderr)

    return expanded

def extract_measures(tree: ET.ElementTree, limit: Optional[int]=None) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    raw = collect_measures(root)
    expanded = expand_repeats_with_volta(raw)

    # LIMIT uygula (opsiyonel)
    if limit is not None and limit >= 0:
        expanded = expanded[:limit]

    linear: List[LinearMeasure] = []
    current_dynamic: Optional[str] = None
    current_tempo: float = 120.0
    current_divisions: int = 1

    global_time_beats: float = 0.0
    global_time_seconds: float = 0.0

    for lin_idx, m in enumerate(expanded):
        lm = LinearMeasure(
            linear_index=lin_idx,
            original_measure=int(m.get('number', '0')) if (m.get('number','0').lstrip('-').isdigit()) else 0
        )

        # Attributes
        attr = m.find(ns.tag('attributes'))
        div_el = attr.find(ns.tag('divisions')) if attr is not None else None
        if div_el is not None:
            try:
                current_divisions = max(1, int(div_el.text))
            except (TypeError, ValueError):
                pass
        divisions = max(current_divisions, 1)

        # Time signature
        beats = None
        beat_type = None
        time_el = attr.find(ns.tag('time')) if attr is not None else None
        if time_el is not None:
            beats_txt = time_el.findtext(ns.tag('beats'))
            beat_type_txt = time_el.findtext(ns.tag('beat-type'))
            if beats_txt and beats_txt.isdigit() and beat_type_txt and beat_type_txt.isdigit():
                beats = int(beats_txt)
                beat_type = int(beat_type_txt)

        # Measure başında direction (tempo/dynamic)
        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic = dyn
                lm.new_dynamic = dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                # Aşırı uç tempo değerlerine karşı min/max sınır (güvenlik)
                if 10.0 <= tmp <= 400.0:
                    current_tempo = tmp
                else:
                    # olağan dışı tempo tespitinde uyar ama yine de clamp et
                    print(f"UYARI: Olağan dışı tempo {tmp} BPM; {max(min(tmp,400.0),10.0)} BPM'e clamp edildi.", file=sys.stderr)
                    current_tempo = max(min(tmp, 400.0), 10.0)
                lm.tempo = current_tempo

        # Ölçü içi akış: position (div) ve maksimum pozisyon
        cur_div_pos = 0
        max_div_pos = 0

        def is_chord(note_el: ET.Element) -> bool:
            return note_el.find(ns.tag('chord')) is not None

        # Çocukları sırayla gez: note/backup/forward/direction
        for child in list(m):
            local = child.tag.split('}', 1)[-1] if '}' in child.tag else child.tag

            if local == 'direction':
                # Ölçü ortasında tempo/dynamic değişebilir (saniyeye çevrimde kaba bir yaklaşım kullanıyoruz)
                dyn = dynamic_from_direction(child, ns)
                if dyn:
                    current_dynamic = dyn
                    lm.new_dynamic = dyn
                tmp = tempo_from_direction(child, ns)
                if tmp:
                    if 10.0 <= tmp <= 400.0:
                        current_tempo = tmp
                    else:
                        print(f"UYARI: Olağan dışı tempo {tmp} BPM (ölçü içi); clamp edildi.", file=sys.stderr)
                        current_tempo = max(min(tmp, 400.0), 10.0)
                    lm.tempo = current_tempo

            elif local == 'backup':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos -= int(dur_txt)
                    if cur_div_pos < 0:
                        cur_div_pos = 0

            elif local == 'forward':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos += int(dur_txt)
                    if cur_div_pos > max_div_pos:
                        max_div_pos = cur_div_pos

            elif local == 'note':
                # Staff filtresi
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

                # Olay zamanı ölçü başlangıcına göre (beat)
                ev_t_beats = global_time_beats + (cur_div_pos / divisions)
                ev_d_beats = (dur_div / divisions)

                if staff == 1:
                    pitches_tr: List[str] = []
                    pitches_sci: List[str] = []
                    midi: List[int] = []
                    if not is_rest and step is not None:
                        pitches_tr.append(turkish_name(step, alter, octave))
                        pitches_sci.append(sci_name(step, alter, octave))
                        midi.append(midi_number(step, alter, octave))

                    # Artikülasyon
                    art: List[str] = []
                    if child.find('.//' + ns.tag('staccato')) is not None: art.append("staccato")
                    if child.find('.//' + ns.tag('accent'))   is not None: art.append("accent")
                    if child.find('.//' + ns.tag('tenuto'))   is not None: art.append("tenuto")
                    if child.find('.//' + ns.tag('strong-accent')) is not None: art.append("marcato")

                    # Slur
                    slur_start = False; slur_end = False
                    for sl in child.findall('.//' + ns.tag('slur')):
                        t = sl.get('type')
                        if t == 'start': slur_start = True
                        elif t == 'stop': slur_end = True

                    ev = NoteEvent(
                        t_beats=ev_t_beats,
                        t_seconds=0.0,               # sonra hesaplanacak
                        dur_beats=ev_d_beats,
                        dur_seconds=0.0,             # sonra hesaplanacak
                        pitches_tr=pitches_tr,
                        pitches_sci=pitches_sci,
                        midi=midi,
                        dyn=current_dynamic,
                        art=art,
                        slur_start=slur_start,
                        slur_end=slur_end,
                        grace=grace,
                    )
                    lm.RH.append(ev)

                # chord olmayan notalarda pozisyonu ilerlet
                if not chord_flag:
                    cur_div_pos += dur_div
                    if cur_div_pos > max_div_pos:
                        max_div_pos = cur_div_pos

        # Ölçü süresi (beats)
        content_beats = max_div_pos / divisions if divisions else 0.0
        ts_beats = None
        if beats and beat_type:
            ts_beats = beats * (4.0 / float(beat_type))

        # Clamp mantığı:
        # - time signature varsa, measure_beats = min(content_beats>0 ? content_beats : ts_beats, ts_beats)
        # - yoksa content_beats
        if ts_beats is not None:
            if content_beats > 0:
                measure_beats = min(content_beats, ts_beats + 1e-9)  # çok küçük tolerans
            else:
                measure_beats = ts_beats
        else:
            measure_beats = content_beats

        # RH olayları için saniye hesabı (ölçü başı tempo ile)
        sec_per_beat = 60.0 / max(current_tempo, 1e-6)
        for ev in lm.RH:
            local_offset = ev.t_beats - global_time_beats
            ev.t_seconds = global_time_seconds + local_offset * sec_per_beat
            ev.dur_seconds = ev.dur_beats * sec_per_beat

        # Global zaman ilerlet
        global_time_seconds += measure_beats * sec_per_beat
        global_time_beats += measure_beats
        lm.measure_beats = measure_beats

        linear.append(lm)

    # Toplam süre çıktı (debug)
    print(f"[INFO] Expanded measures: {len(expanded)} (raw: {len(raw)})", file=sys.stderr)
    print(f"[INFO] Total duration (s): {global_time_seconds:.3f}", file=sys.stderr)

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
        "MusicXML Transcript (RH only, repeat+volta expanded, TS-clamped)",
        f"Generated UTC: {now}",
        f"Linear measures: {len(measures)}",
        f"Events: {len(events)}",
        "---",
        "time_sec\tdur_sec\tpitches_sci\tdyn\tart"
    ]
    for e in events:
        txt.append(f"{e['t']:.3f}\t{e['dur']:.3f}\t{','.join(e['pitches_sci']) or 'rest'}\t{e['dyn'] or ''}\t{','.join(e['art'])}")
    (out_dir/"transcript_full.txt").write_text("\n".join(txt), encoding="utf-8")

    # JSON
    json_obj = {
        'metadata': {
            'generated_utc': now,
            'measure_count': len(measures),
            'event_count': len(events),
            'format_version': 7,
            'script_version': __VERSION__,
        },
        'events': events
    }
    (out_dir/"transcript_full.json").write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # ABC placeholder
    abc_lines = ["X:1", "T:Extracted RH", "M:4/4", "L:1/16", "Q:1/4=120", "K:C", "V:1 clef=treble"]
    (out_dir/"transcript_full.abc").write_text("\n".join(abc_lines), encoding="utf-8")

    # Measure map
    map_lines = ["# Measure Map", "",
                 "| LinearIndex | OriginalMeasure | RH_events | MeasureBeats |",
                 "|------------:|---------------:|----------:|-------------:|"]
    for m in measures:
        map_lines.append(f"| {m.linear_index} | {m.original_measure} | {len(m.RH)} | {m.measure_beats:.3f} |")
    (out_dir/"measure_map.md").write_text("\n".join(map_lines), encoding="utf-8")

    # Lua exporter
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
        "-- Auto-generated RH transcription table (repeat+volta expanded, TS-clamped)",
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
    (out_dir/"transcript_full.lua").write_text("\n".join(lua_lines), encoding="utf-8")

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
              "Çıktılar: transcript_full.txt, .json, .lua, .abc, .md\n", file=sys.stderr)
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
