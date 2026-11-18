from src.answer import Citation, detect_language, extract_citations, has_script_leak, is_refusal


def test_detect_language_arabic():
    assert detect_language("ما هي درجة غليان النيتروجين السائل؟") == "ar"


def test_detect_language_english():
    assert detect_language("What is the boiling point of liquid nitrogen?") == "en"


def test_extract_citations_single():
    text = "Liquid nitrogen is cold. [safetygram-7-liquid-nitrogen, page 2]"
    assert extract_citations(text) == [Citation(document="safetygram-7-liquid-nitrogen", page=2)]


def test_extract_citations_multiple_deduplicated():
    text = (
        "A. [doc-a, page 1] B. [doc-b, page 3] C. [doc-a, page 1]"
    )
    citations = extract_citations(text)
    assert citations == [
        Citation(document="doc-a", page=1),
        Citation(document="doc-b", page=3),
    ]


def test_is_refusal_detects_english_and_arabic():
    assert is_refusal("Not covered in the provided documents")
    assert is_refusal("هذا غير مذكور في الوثائق المتوفرة")
    assert not is_refusal("Liquid nitrogen is extremely cold. [doc, page 1]")


def test_has_script_leak_ignores_citation_brackets():
    text = "النيتروجين السائل بارد جدًا. [safetygram-7-liquid-nitrogen, page 1]"
    assert not has_script_leak(text)


def test_has_script_leak_detects_latin_and_cjk_in_body():
    assert has_script_leak("النيتروgen السائل بارد جدًا.")
    assert has_script_leak("当处理液化氢气罐时، يجب الحذر.")
