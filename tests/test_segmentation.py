from messina_info.segmentation import LanguageSection, split_language_sections


RU = "\U0001f1f7\U0001f1fa"
EN = "\U0001f1ec\U0001f1e7"
FOOTER = "t.me/MessinaInfo | instagram.com/messinainfo"


def _post(russian: str, english: str, title: str = "Shared title") -> str:
    return f"{title}\n\n{RU}\n\n{russian}\n\n—\n\n{EN}\n\n{english}\n\n{FOOTER}"


def test_standard_post_produces_russian_and_english_sections() -> None:
    sections = split_language_sections(
        _post(
            "Сегодня университет опубликовал важное объявление.",
            "Today the university published an important announcement.",
        )
    )

    assert [section.language for section in sections] == ["ru", "en"]
    assert sections[0].text.endswith("Сегодня университет опубликовал важное объявление.")
    assert sections[1].text.endswith(
        "Today the university published an important announcement."
    )


def test_shared_title_and_content_before_marker_are_in_both_sections() -> None:
    title = "Important date: 12 September"
    sections = split_language_sections(
        _post(
            "Подробное описание события для русскоязычных читателей.",
            "A detailed description of the event for English readers.",
            title,
        )
    )

    assert sections == [
        LanguageSection(
            "ru",
            title,
            f"{title}\n\nПодробное описание события для русскоязычных читателей.",
        ),
        LanguageSection(
            "en",
            title,
            f"{title}\n\nA detailed description of the event for English readers.",
        ),
    ]


def test_footer_removed_but_external_source_url_preserved() -> None:
    source = "Источник: https://example.org/notizie/important-update"
    sections = split_language_sections(
        _post(
            f"Подробности доступны здесь.\n\n{source}",
            "Full details are available from the external source above.",
        )
    )

    assert source in sections[0].text
    assert all(FOOTER not in section.text for section in sections)


def test_separator_removed_but_internal_em_dash_preserved() -> None:
    russian = "Лекция — открытая для всех студентов университета."
    sections = split_language_sections(
        _post(russian, "The lecture is open to all university students.")
    )

    assert russian in sections[0].text
    assert "\n—\n" not in sections[0].text


def test_missing_markers_returns_one_undetermined_section() -> None:
    text = "Event notice\n\nThe university event starts on 15 September."

    assert split_language_sections(text) == [LanguageSection("und", "", text)]


def test_empty_language_block_is_skipped() -> None:
    sections = split_language_sections(
        _post("", "This English language block contains enough useful information.")
    )

    assert [section.language for section in sections] == ["en"]


def test_short_non_informative_text_is_skipped() -> None:
    assert split_language_sections("ESPAÑA 🇪🇸") == []


def test_excessive_blank_lines_are_normalized() -> None:
    sections = split_language_sections(
        _post(
            "Первый содержательный абзац.\n\n\n\nВторой содержательный абзац.",
            "First informative paragraph.\n\n\n\nSecond informative paragraph.",
        )
    )

    assert "\n\n\n" not in sections[0].text
    assert "Первый содержательный абзац.\n\nВторой" in sections[0].text


def test_repeated_calls_are_identical_and_input_is_unchanged() -> None:
    text = _post(
        "Содержательный русский текст для проверки результата.",
        "Meaningful English text used to verify deterministic output.",
    )
    original = text[:]

    first = split_language_sections(text)
    second = split_language_sections(text)

    assert first == second
    assert text == original
