#!/usr/bin/env python3
"""
MusicXML'den SADECE sağ el (RH, staff=1) notalarını çıkaran,
eşleştirilmiş repeat (|: :|) + volta (1.,2. ending) + D.C./D.S./To Coda/Fine
işaretlerini deterministik şekilde uygulayan transcript scripti.

Yenilikler:
- D.C./D.S./To Coda/Fine için <direction-type><words> metinleri de taranır
  (ör. "D.S. al Coda", "To Coda", "Da Capo", "Fine", "Segno", "Coda").
- Global repeat pas sayacı: Her (start,end) çifti, tüm playback boyunca
  en fazla 'times' kadar uygulanır (D.C./D.S. dönüşünden sonra sınırsız tekrar
  döngülerini engeller). İsterseniz tekrarları dönüşte yeniden uygulamak için
  REPEAT_REAPPLY_ON_RETURN=1 yapabilirsiniz.
- playback_order.txt ile ölçü yürüyüş izini dosyaya yazar.
- Sonsuz döngü koruması: playback state dedup + makul adım sınırı.

USAGE:
  python scripts/generate_transcript.py <input.musicxml> [LIMIT]
    LIMIT: all/-1 => tümü; pozitif sayı => ilk N ölçü (repeat açılımından sonra)

Outputs:
  transcript_rh.txt, transcript_rh.json, transcript_rh.lua, transcript_rh.abc, measure_map_rh.md, playback_order.txt
"""
import os
import sys
import json
import datetime
import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any

__VERSION__ = "1.1.0"

DYNAMIC_TAGS = ["ppp","pp","p","mp","mf","f","ff","fff","fp","sfz"]
LETTER_TO_TURKISH = {"C":"Do","D":"Re","E":"Mi","F":"Fa","G":"Sol","A":"La","B":"Si"}
LETTER_TO_SEMITONE = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

REPEAT_REAPPLY_ON_RETURN = os.getenv("REPEAT_REAPPLY_ON_RETURN", "0").strip() == "1"

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
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return out

    for idx, m in enumerate(measures):
        for barline in m.findall('.//' + ns.tag('barline')):
            ending = barline.find(ns.tag('ending'))
            if ending is None: continue
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
        for barline in m.findall('.//' + ns.tag('barline')):
            repeat = barline.find(ns.tag('repeat'))
            if repeat is None:
                continue
            direction = repeat.get('direction')
            if direction == 'forward':
                stack.append(i)
            elif direction == 'backward':
                if not stack:
                    continue
                start = stack.pop()
                t = repeat.get('times')
                times = int(t) if (t and t.isdigit()) else 2
                start_to_pair[start] = (i, times)
                end_to_pair[i] = (start, times)
    return start_to_pair, end_to_pair

# basit normalize
_word_re = re.compile(r"[^\w]+", re.UNICODE)
def norm_words(s: str) -> str:
    s = s.strip().lower()
    s = _word_re.sub(" ", s)
    return " ".join(s.split())

def scan_markers(measures: List[ET.Element]) -> Dict[str, Any]:
    if not measures:
        return {}
    ns = NS(measures[0])

    has_segno: Set[int] = set()
    has_coda: Set[int] = set()
    has_fine: Set[int] = set()
    trig_dc: Set[int] = set()
    trig_ds: Set[int] = set()
    trig_tocoda: Set[int] = set()

    for i, m in enumerate(measures):
        for d in m.findall(ns.tag('direction')):
            # explicit segno/coda elements
            if d.find(ns.tag('segno')) is not None:
                has_segno.add(i)
            if d.find(ns.tag('coda')) is not None:
                has_coda.add(i)

            # words: D.C., Da Capo, D.S., To Coda, Fine, Segno, Coda
            dtyp = d.find(ns.tag('direction-type'))
            if dtyp is not None:
                for w in dtyp.findall(ns.tag('words')):
                    txt = norm_words(w.text or "")
                    if not txt:
                        continue
                    if txt in ("d c", "da capo", "da capo al fine", "da capo al coda"):
                        trig_dc.add(i)
                    if txt.startswith("d s") or "dal segno" in txt or "del segno" in txt:
                        trig_ds.add(i)
                    if "to coda" in txt or txt == "coda":
                        trig_tocoda.add(i)
                    if txt == "fine":
                        has_fine.add(i)
                    if txt == "segno":
                        has_segno.add(i)

            # sound attributes (MuseScore genellikle ekler)
            sound = d.find(ns.tag('sound'))
            if sound is not None:
                if sound.get('fine') is not None:
                    has_fine.add(i)
                if sound.get('dacapo') is not None:
                    trig_dc.add(i)
                if sound.get('dalsegno') is not None:
                    trig_ds.add(i)
                if sound.get('tocoda') is not None:
                    trig_tocoda.add(i)

    return {
        'ending_membership': parse_endings(measures, ns),
        'start_to_pair': pair_repeats(measures)[0],
        'end_to_pair': pair_repeats(measures)[1],
        'has_segno': has_segno,
        'has_coda': has_coda,
        'has_fine': has_fine,
        'trig_dc': trig_dc,
        'trig_ds': trig_ds,
        'trig_tocoda': trig_tocoda,
    }

@dataclass(frozen=True)
class FrameState:
    start: int
    end: int
    pass_no: int
    max_times: int

def expand_playback(measures: List[ET.Element]) -> List[int]:
    """
    Repeat + Volta + DC/DS/To Coda/Fine çalma sırası.
    - DC/DS her biri en fazla bir kez.
    - To Coda dönüş sonrasında bir kez.
    - Global repeat sayaçları: Her (start,end) çifti toplamda 'times' kadar.
      REPEAT_REAPPLY_ON_RETURN=1 ise dönüşlerde tekrar yeniden uygulanır.
    """
    n = len(measures)
    if n == 0:
        return []
    ns = NS(measures[0])

    mk = scan_markers(measures)
    ending_membership: Dict[int, Tuple[int, Set[int]]] = mk['ending_membership']
    start_to_pair: Dict[int, Tuple[int,int]] = mk['start_to_pair']
    end_to_pair: Dict[int, Tuple[int,int]] = mk['end_to_pair']
    has_segno: Set[int] = mk['has_segno']
    has_coda: Set[int] = mk['has_coda']
    has_fine: Set[int] = mk['has_fine']
    trig_dc: Set[int] = mk['trig_dc']
    trig_ds: Set[int] = mk['trig_ds']
    trig_tocoda: Set[int] = mk['trig_tocoda']

    segno_idx = min(has_segno) if has_segno else None
    coda_idx = min(has_coda) if has_coda else None

    playback: List[int] = []
    i = 0
    frames: List[Tuple[int,int,int,int]] = []  # (start, end, pass_no, max_times)

    used_dc = False
    used_ds = False
    jumped_coda = False

    # Global repeat kullanım sayacı
    global_repeat_count: Dict[Tuple[int,int], int] = {}

    visited: Set[Tuple[int, Tuple[FrameState, ...], bool, bool, bool]] = set()

    raw_n = n
    max_steps = max(1500, raw_n * 8)  # güvenlik: orijinalin ~8 katı
    steps = 0
    DEBUG = os.getenv('DEBUG_PLAYBACK', '0') == '1'

    while 0 <= i < n and steps < max_steps:
        steps += 1

        # Volta ending uygunsuzsa o bloğu atla
        if frames:
            end_tup = ending_membership.get(i)
            if end_tup:
                end_idx, ending_nums = end_tup
                s, e, pass_no, max_times = frames[-1]
                if ending_nums and (pass_no not in ending_nums):
                    i = end_idx + 1
                    continue

        fs = tuple(FrameState(*f) for f in frames)
        state = (i, fs, used_dc, used_ds, jumped_coda)
        if state in visited:
            # Aynı bağlam tekrarlandığında çalmayı bitir (döngü)
            print(f"UYARI: Playback state tekrarlandı, çalma sonlandırıldı. i={i}, frames={fs}, dc={used_dc}, ds={used_ds}, coda={jumped_coda}", file=sys.stderr)
            break
        visited.add(state)

        playback.append(i)

        # Repeat başlangıcı
        if i in start_to_pair:
            end_idx, times = start_to_pair[i]
            frames.append((i, end_idx, 1, max(1, times)))

        # Repeat bitişi
        if i in end_to_pair and frames:
            s, e, pass_no, max_times = frames[-1]
            start_idx, times_b = end_to_pair[i]
            if s == start_idx and e == i:
                if times_b:
                    max_times = times_b
                # Global sayaç kontrolü
                pair_key = (s, e)
                cur_used = global_repeat_count.get(pair_key, 0)
                # Dönüşte repeat tekrar açılsın mı?
                allowed_times_total = max_times if REPEAT_REAPPLY_ON_RETURN else max_times
                # Eğer dönüşte tekrar açılmasını istemiyorsanız, global toplamı max_times ile sınırlarız:
                if not REPEAT_REAPPLY_ON_RETURN:
                    if cur_used >= allowed_times_total - 1:  # ilk çalım + (times-1) tekrar
                        frames.pop()
                    else:
                        if pass_no < max_times:
                            frames[-1] = (s, e, pass_no + 1, max_times)
                            global_repeat_count[pair_key] = cur_used + 1
                            i = s
                            continue
                        else:
                            frames.pop()
                else:
                    # Dönüşte yeniden uygulama açık ise, sadece lokal çerçeveye göre karar ver
                    if pass_no < max_times:
                        frames[-1] = (s, e, pass_no + 1, max_times)
                        global_repeat_count[pair_key] = cur_used + 1
                        i = s
                        continue
                    else:
                        frames.pop()

        # Fine: DS/DC dönüşünden sonra Fine'da bitir
        if i in has_fine and (used_dc or used_ds):
            break

        # D.S.
        did_jump = False
        if (i in trig_ds) and (not used_ds) and (segno_idx is not None):
            used_ds = True
            frames.clear()
            i = segno_idx
            did_jump = True

        # D.C.
        if (not did_jump) and (i in trig_dc) and (not used_dc):
            used_dc = True
            frames.clear()
            i = 0
            did_jump = True

        # To Coda (yalnız dönüş sonrası)
        if (not did_jump) and (i in trig_tocoda) and (coda_idx is not None) and (used_dc or used_ds) and (not jumped_coda):
            jumped_coda = True
            frames.clear()
            i = coda_idx
            did_jump = True

        if did_jump:
            continue

        i += 1

    if steps >= max_steps:
        print(f"UYARI: Playback açılımı güvenlik sınırında durduruldu ({steps} adım). İşaretleri kontrol edin.", file=sys.stderr)

    if len(playback) > raw_n * 8:
        print(f"UYARI: Playback uzunluğu sıra dışı: {raw_n} -> {len(playback)} ölçü.", file=sys.stderr)

    # Debug: playback_order.txt yaz
    try:
        with open("playback_order.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in playback))
    except Exception:
        pass

    return playback

def extract_measures(tree: ET.ElementTree, limit: Optional[int]=None) -> List[LinearMeasure]:
    root = tree.getroot()
    ns = NS(root)
    raw = collect_measures(root)
    order = expand_playback(raw)

    if limit is not None and limit >= 0:
        order = order[:limit]

    linear: List[LinearMeasure] = []
    current_dynamic: Optional[str] = None
    current_tempo: float = 120.0
    current_divisions: int = 1

    global_time_beats: float = 0.0
    global_time_seconds: float = 0.0

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

        # Time signature -> measure beats
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
            # İçerikten tahmin (fallback)
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
                    chord_tag = child.find(ns.tag('chord'))
                    if chord_tag is None:
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
                if tmp < 10.0 or tmp > 400.0:
                    print(f"UYARI: Olağan dışı tempo {tmp} BPM; clamp edildi.", file=sys.stderr)
                current_tempo = min(max(tmp, 10.0), 400.0)
                lm.tempo = current_tempo

        # RH olayları: ölçü içi akış
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
    if len(order) > len(raw) * 8:
        print(f"UYARI: Playback açılımı sıra dışı uzun: {len(raw)} -> {len(order)}.", file=sys.stderr)

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
        "MusicXML Sağ El Transcript (RH only, repeat+volta+DC/DS/Coda, TS-based)",
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
            'format_version': 11,
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
        "-- Auto-generated RH transcription table (repeat+volta+DC/DS/Coda, TS-based)",
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

    print(f"\\n[INFO] Wrote {len(events)} RH events across {len(measures)} measures.")
    # playback_order.txt zaten expand_playback içinde yazıldı.

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
              "Çıktılar: transcript_rh.txt, .json, .lua, .abc, .md, playback_order.txt\n", file=sys.stderr)
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
