#!/usr/bin/env python3
"""
MusicXML'den SADECE sağ el (RH, staff=1) notalarını çıkaran,
repeat (ileri/geri) ve volta (1.,2. son) işaretlerini açan ve
ölçü içi akışı (note/backup/forward/chord) doğru takip eden transcript scripti.

- Müziğin gerçek çalınış sırasını bozmadan tekrarları genişletir (|: ... :| ve 1.,2. ending).
- RH dışındaki içerik ZAMAN AKIŞINI belirlemek için dikkate alınır (ölçü süresi doğru ilerler),
  ama çıktıya sadece RH olayları yazılır.
- D.C./D.S./Coda/Fine için uyarı verir (tam simülasyon opsiyonel olarak eklenebilir).

USAGE:
    python scripts/generate_transcript.py <input.musicxml>
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

__VERSION__ = "0.6.0"

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
    tempo: Optional[float] = None
    new_dynamic: Optional[str] = None

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
    # MusicXML'de tempo çoğunlukla <direction><sound tempo="..."> ile gelir
    sound = direction_el.find(ns.tag('sound')) if direction_el is not None else None
    if sound is not None and sound.get('tempo'):
        try:
            return float(sound.get('tempo'))
        except ValueError:
            return None
    return None

def collect_measures(root: ET.Element) -> List[ET.Element]:
    ns = NS(root)
    part = root.find(ns.tag('part'))
    if part is None:
        return []
    return list(part.findall(ns.tag('measure')))

def parse_ending_ranges(measures: List[ET.Element], ns: NS) -> Dict[int, Tuple[int, Set[int]]]:
    """
    Ending kapsamlarını çıkar. Sonuç: ending_membership[i] = (end_idx, {1,2,...})
    """
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
    """
    İleri/geri repeat ve ending (volta) destekli açılım.
    DC/DS/Coda/Fine bu sürümde sadece uyarı verir.
    """
    if not measures:
        return []
    ns = NS(measures[0])
    ending_membership = parse_ending_ranges(measures, ns)

    expanded: List[ET.Element] = []
    i = 0

    # Repeat frame yapısı: (start_idx, end_idx_or_None, pass_no, max_times)
    frames: List[Tuple[int, Optional[int], int, int]] = []

    safety = 0
    while i < len(measures):
        safety += 1
        if safety > 100000:
            print("UYARI: Repeat açılımında güvenlik sınırı aşıldı, olası sonsuz döngü.", file=sys.stderr)
            break

        m = measures[i]

        # Karmaşık tekrar işaretleri için uyarı
        for direction in m.findall(ns.tag('direction')):
            sound = direction.find(ns.tag('sound'))
            if sound is not None and (sound.get('dacapo') or sound.get('dalsegno') or sound.get('fine') or sound.get('tocoda') or sound.get('coda') or sound.get('segno')):
                print(f"UYARI: DC/DS/Coda/Fine algılandı (ölçü {m.get('number','?')}); bu sürüm sadece ileri/geri repeat ve ending açar.", file=sys.stderr)

        # Volta (ending) kontrolü: Eğer bir repeat çerçevesi içindeysek ve bu ending mevcut pass'a uymuyorsa atla
        if frames:
            end_tup = ending_membership.get(i)
            if end_tup:
                end_idx, ending_nums = end_tup
                start_idx, end_idx_frame, pass_no, max_times = frames[-1]
                # ending_nums boş ise her passta çalınır varsayalım
                if ending_nums and (pass_no not in ending_nums):
                    i = end_idx + 1
                    continue

        expanded.append(m)

        # Barline repeat işleme
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
                    backward_times = int(t) if t and t.isdigit() else None

        if forward_here:
            frames.append((i, None, 1, 2))  # varsayılan 2 kez

        if backward_here:
            if frames:
                start_idx, _, pass_no, max_times = frames[-1]
                # times belirtilmişse kullan
                if backward_times is not None:
                    max_times = backward_times
                if pass_no < max_times:
                    frames[-1] = (start_idx, i, pass_no + 1, max_times)
                    i = start_idx  # tekrar başına dön
                    continue
                else:
                    # Çerçeveyi kapat ve devam et
                    frames.pop()

        i += 1

    return expanded

def extract_measures(tree: ET.ElementTree) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    raw = collect_measures(root)
    expanded = expand_repeats_with_volta(raw)

    linear: List[LinearMeasure] = []
    current_dynamic: Optional[str] = None
    current_tempo: float = 120.0
    current_divisions: int = 1

    # Global zaman (beats ve seconds) ölçü başlarında ilerletilir
    global_time_beats: float = 0.0
    global_time_seconds: float = 0.0

    for lin_idx, m in enumerate(expanded):
        no_txt = m.get('number', '0')
        try:
            meas_no = int(no_txt)
        except ValueError:
            meas_no = 0
        lm = LinearMeasure(linear_index=lin_idx, original_measure=meas_no)

        # Ölçü başı attributes
        attr = m.find(ns.tag('attributes'))
        div_el = attr.find(ns.tag('divisions')) if attr is not None else None
        if div_el is not None and (div_el.text or "").isdigit():
            current_divisions = int(div_el.text)
        divisions = max(current_divisions, 1)

        # Ölçü başında görülen direction'lar (tempo/dynamic)
        for direction in m.findall(ns.tag('direction')):
            dyn = dynamic_from_direction(direction, ns)
            if dyn:
                current_dynamic = dyn
                lm.new_dynamic = dyn
            tmp = tempo_from_direction(direction, ns)
            if tmp:
                current_tempo = tmp
                lm.tempo = tmp

        # Ölçü içi akış: note/backup/forward/direction sırasını korumak için çocukları sırayla dolaş
        cur_div_pos = 0  # ölçü içi pozisyon (divisions biriminde)
        max_div_pos = 0

        # Chord notalarının süre eklememesi için kontrol
        def is_chord(note_el: ET.Element) -> bool:
            return note_el.find(ns.tag('chord')) is not None

        for child in list(m):
            tag = child.tag if ns.ns is None else child.tag
            local = tag.split('}', 1)[-1] if '}' in tag else tag

            if local == 'direction':
                # Ölçü içinde tempo/dynamic değişebilir; beats tabanlı zaman çizgimiz değişmez,
                # ama yine de bir sonraki olaylar için tempo/dynamic güncelleyelim.
                dyn = dynamic_from_direction(child, ns)
                if dyn:
                    current_dynamic = dyn
                    lm.new_dynamic = dyn
                tmp = tempo_from_direction(child, ns)
                if tmp:
                    current_tempo = tmp
                    lm.tempo = tmp

            elif local == 'backup':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos -= int(dur_txt)
                    if cur_div_pos < 0:
                        cur_div_pos = 0  # güvenlik
                # max_div_pos değişmez

            elif local == 'forward':
                dur_txt = child.findtext(ns.tag('duration'))
                if dur_txt and dur_txt.isdigit():
                    cur_div_pos += int(dur_txt)
                    if cur_div_pos > max_div_pos:
                        max_div_pos = cur_div_pos

            elif local == 'note':
                # RH filtresi
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

                # Olay zamanı (beats/saniye) ölçü başına global offset ile hesaplanır
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

                    # Basit artikülasyonlar
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
                        t_seconds=0.0,               # Geçici, ölçü sonunda güncellenecek
                        dur_beats=ev_d_beats,
                        dur_seconds=0.0,             # Geçici, ölçü sonunda güncellenecek
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

                # Zaman pozisyonu: chord notasıysa ilerleme yok; değilse duration kadar ilerle
                if not chord_flag:
                    cur_div_pos += dur_div
                    if cur_div_pos > max_div_pos:
                        max_div_pos = cur_div_pos

            else:
                # diğer tag'lar yok sayılır
                pass

        # Ölçü süresi (beats) = max_div_pos / divisions
        measure_beats = max_div_pos / divisions

        # Bu ölçüdeki RH olayları için t_seconds ve dur_seconds hesapla
        # Basitleştirme: ölçü boyunca tek tempo varsayıyoruz (ölçü başı tempo)
        for ev in lm.RH:
            local_offset_beats = ev.t_beats - global_time_beats  # ölçü içi offset
            ev.t_seconds = global_time_seconds + local_offset_beats * (60.0 / max(current_tempo, 1e-9))
            ev.dur_seconds = ev.dur_beats * (60.0 / max(current_tempo, 1e-9))

        # Bir sonraki ölçüye global zamanı ilerlet
        global_time_seconds += measure_beats * (60.0 / max(current_tempo, 1e-9))
        global_time_beats += measure_beats

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
        "MusicXML Sağ El Transcript (RH only, repeat+volta expanded, backup/forward aware)",
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
            'format_version': 6,
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
        "-- Auto-generated RH transcription table (repeat+volta expanded, backup/forward aware)",
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

    print(f"Wrote {len(events)} RH events across {len(measures)} expanded measures.")

def main():
    if len(sys.argv) < 2:
        print("USAGE: python scripts/generate_transcript.py <input.musicxml>\n"
              "Sadece sağ el (RH) çıkarılır. Repeat+volta açılır; backup/forward akışı dikkate alınır.\n"
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
