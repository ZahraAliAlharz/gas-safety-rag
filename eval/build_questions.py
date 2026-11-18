"""Generate eval/questions.csv: 10 English questions + their Arabic translations
(20 rows total), 4 of each 10 deliberately unanswerable from the corpus.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REFUSAL_EN = "Not covered in the provided documents"
REFUSAL_AR = "هذا غير مذكور في الوثائق المتوفرة"

QUESTIONS = [
    dict(
        num=1,
        answerable=True,
        en="What is the boiling point of liquid nitrogen?",
        ar="ما هي درجة غليان النيتروجين السائل؟",
        gold_answer="-320.5°F (-195.8°C) at 1 atm",
        gold_doc="safetygram-7-liquid-nitrogen",
        gold_page=2,
    ),
    dict(
        num=2,
        answerable=True,
        en="What is the flammable range of gaseous hydrogen in air?",
        ar="ما هو نطاق الاشتعال للهيدروجين الغازي في الهواء؟",
        gold_answer="4% to 75% by volume",
        gold_doc="safetygram-4-gaseous-hydrogen",
        gold_page=1,
    ),
    dict(
        num=3,
        answerable=True,
        en="What first aid should be given for skin contact with liquid oxygen?",
        ar="ما هي الإسعافات الأولية التي يجب تقديمها في حالة تلامس الجلد مع الأكسجين السائل؟",
        gold_answer=(
            "Remove restrictive clothing, do not rub the frozen area, place it in a warm "
            "water bath not exceeding 105°F (40°C), never use dry heat, and call a physician"
        ),
        gold_doc="safetygram-6-liquid-oxygen",
        gold_page=8,
    ),
    dict(
        num=4,
        answerable=True,
        en=(
            "What minimum oxygen concentration is required for personnel to work without "
            "supplied air, according to the liquid nitrogen Safetygram?"
        ),
        ar=(
            "ما هو الحد الأدنى لتركيز الأكسجين المطلوب للعاملين للعمل دون هواء معاد تزويده، "
            "وفقًا لنشرة السلامة الخاصة بالنيتروجين السائل؟"
        ),
        gold_answer="19.5% oxygen concentration",
        gold_doc="safetygram-7-liquid-nitrogen",
        gold_page=6,
    ),
    dict(
        num=5,
        answerable=True,
        en="What PPE is recommended when handling gaseous hydrogen cylinders?",
        ar="ما هي معدات الحماية الشخصية الموصى بها عند التعامل مع أسطوانات الهيدروجين الغازي؟",
        gold_answer=(
            "Flame retardant protective clothing, safety glasses, safety shoes, hard hats, "
            "personal gas monitors, and sturdy work gloves"
        ),
        gold_doc="safetygram-4-gaseous-hydrogen",
        gold_page=6,
    ),
    dict(
        num=6,
        answerable=True,
        en=(
            "What is the minimum safe distance that electrical equipment must maintain from a "
            "gaseous hydrogen storage vessel?"
        ),
        ar=(
            "ما هي المسافة الآمنة الدنيا التي يجب أن تحافظ عليها المعدات الكهربائية من وعاء "
            "تخزين الهيدروجين الغازي؟"
        ),
        gold_answer="15 feet (4 meters)",
        gold_doc="safetygram-4-gaseous-hydrogen",
        gold_page=5,
    ),
    dict(
        num=7,
        answerable=False,
        en="What is the current retail price per liter of liquid nitrogen?",
        ar="ما هو السعر الحالي بالتجزئة لليتر الواحد من النيتروجين السائل؟",
        gold_answer=REFUSAL_EN,
        gold_doc="",
        gold_page="",
    ),
    dict(
        num=8,
        answerable=False,
        en="Does Air Products provide a mobile app for remotely monitoring cryogenic tank levels?",
        ar="هل توفر شركة Air Products تطبيقًا للهاتف المحمول لمراقبة مستويات خزانات الغازات المبردة عن بُعد؟",
        gold_answer=REFUSAL_EN,
        gold_doc="",
        gold_page="",
    ),
    dict(
        num=9,
        answerable=False,
        en="What is the carbon footprint (CO2 emissions) of producing one ton of liquid oxygen?",
        ar="ما هو الأثر الكربوني (انبعاثات ثاني أكسيد الكربون) لإنتاج طن واحد من الأكسجين السائل؟",
        gold_answer=REFUSAL_EN,
        gold_doc="",
        gold_page="",
    ),
    dict(
        num=10,
        answerable=False,
        en="What is the maximum allowable truck transport distance for liquid hydrogen under DOT regulations?",
        ar=(
            "ما هي أقصى مسافة نقل مسموح بها بالشاحنات للهيدروجين السائل بموجب لوائح وزارة النقل "
            "الأمريكية (DOT)؟"
        ),
        gold_answer=REFUSAL_EN,
        gold_doc="",
        gold_page="",
    ),
]


def build_rows() -> list[dict]:
    rows = []
    for q in QUESTIONS:
        gold_answer_ar = REFUSAL_AR if not q["answerable"] else q["gold_answer"]
        rows.append(
            dict(
                id=f"en-{q['num']:02d}",
                lang="en",
                question=q["en"],
                answerable="yes" if q["answerable"] else "no",
                gold_answer=q["gold_answer"],
                gold_doc=q["gold_doc"],
                gold_page=q["gold_page"],
            )
        )
        rows.append(
            dict(
                id=f"ar-{q['num']:02d}",
                lang="ar",
                question=q["ar"],
                answerable="yes" if q["answerable"] else "no",
                gold_answer=gold_answer_ar,
                gold_doc=q["gold_doc"],
                gold_page=q["gold_page"],
            )
        )
    return rows


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "questions.csv"
    df = pd.DataFrame(build_rows())
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} questions -> {out_path}")
    print(df["answerable"].value_counts())
