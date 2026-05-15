from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "lab1_news_titles.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

UKRAINIAN_STOP_WORDS = {
    "а", "аби", "або", "але", "аж", "ба", "без", "би", "був", "була", "були",
    "було", "бути", "в", "вам", "вами", "вас", "ваш", "ваша", "ваше", "ваші",
    "вже", "ви", "від", "він", "вона", "вони", "воно", "все", "всім", "всіх",
    "де", "для", "до", "дуже", "ж", "за", "з", "зі", "із", "й", "її", "їм",
    "їх", "і", "коли", "коло", "крізь", "лише", "мене", "мені", "ми", "мій",
    "моя", "моє", "мої", "на", "над", "нам", "нами", "нас", "наш", "наша",
    "наше", "не", "нема", "неначе", "ні", "ну", "о", "об", "от", "перед",
    "під", "по", "при", "про", "свій", "своя", "своє", "свої", "се", "та",
    "так", "таки", "там", "те", "тебе", "тобі", "то", "той", "тою", "тут",
    "ти", "у", "усе", "усі", "хай", "хоч", "це", "цей", "ця", "ці", "чи",
    "ще", "що", "щоб", "я", "як", "яка", "який", "які",
}

UKRAINIAN_STOP_WORDS.update({
    "бо", "їй", "його", "ой", "тепер", "тільки", "хто", "чого",
    "десь", "кудись", "знов", "знову", "такий", "таке", "така",
    "отак", "отже", "хіба", "адже", "наче", "мов", "немов",
    "нього", "ній", "ним", "собі", "себе", "же", "потім", "вже",
    "знаю", "може", "буде", "була", "були", "було",
    "неї", "уже", "треба", "виходить", "щось", "йому", "тута", "сам",
})

UKRAINIAN_STOP_WORDS.update({
    "щодо", "чт", "пн", "вт", "ср", "пт", "сб", "нд",
    "сьогодні", "березня", "години",
})

MANUAL_LEMMAS = {
    "лісова": "лісовий",
    "лісової": "лісовий",
    "лісову": "лісовий",
    "ліс": "ліс",
    "пісня": "пісня",
    "пісні": "пісня",
    "мавка": "мавка",
    "мавки": "мавка",
    "мавко": "мавка",
    "лукаш": "лукаш",
    "лукаша": "лукаш",
    "лев": "лев",
    "лева": "лев",
    "мати": "мати",
    "матері": "мати",
    "русалка": "русалка",
    "русалки": "русалка",
    "перелесник": "перелесник",
    "водяник": "водяник",
}

UKRAINIAN_SUFFIXES = (
    "уваннями", "уванням", "ування", "остями", "остях", "ості", "істю",
    "еві", "ові", "ами", "ями", "ого", "ему", "ими", "ими", "ій", "ий",
    "ої", "ою", "ею", "ах", "ях", "ам", "ям", "ів", "їв", "ся", "сь",
    "ти", "ла", "ли", "ло", "на", "не", "ні", "а", "я", "у", "ю", "и",
    "і", "е", "о",
)


@dataclass
class PipelineResult:
    name: str
    available: bool
    elapsed_ms: float
    token_count: int
    unique_count: int
    without_stop_count: int
    top_10: list[tuple[str, int]]
    note: str


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if "clean_title" in (reader.fieldnames or []):
                return ".\n".join(row["clean_title"] for row in reader if row.get("clean_title"))
            if "title" in (reader.fieldnames or []):
                return ".\n".join(row["title"] for row in reader if row.get("title"))
        raise ValueError(f"CSV file {path} must contain 'clean_title' or 'title' column.")
    return path.read_text(encoding="utf-8")


def normalize_apostrophe(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("´", "'")
    )


def filter_noise(text: str) -> str:
    text = normalize_apostrophe(text)
    text = text.replace("\ufeff", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^А-Яа-яІіЇїЄєҐґA-Za-z'\-\s.!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = filter_noise(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def regex_word_tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яіїєґa-z]+(?:['-][а-яіїєґa-z]+)?", text.lower(), flags=re.IGNORECASE)


def whitespace_tokenize(text: str) -> list[str]:
    punctuation = ".,!?;:()[]{}\""
    return [token.strip(punctuation).lower() for token in text.split() if token.strip(punctuation)]


def translate_table_tokenize(text: str) -> list[str]:
    punctuation = ".,!?;:()[]{}\"«»"
    table = str.maketrans({symbol: " " for symbol in punctuation})
    prepared = text.translate(table)
    return [token.lower() for token in prepared.split() if token]


def sentence_tokenize(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def remove_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in UKRAINIAN_STOP_WORDS and len(token) > 1]


def simple_ukrainian_lemma(word: str) -> str:
    if word in MANUAL_LEMMAS:
        return MANUAL_LEMMAS[word]
    for suffix in UKRAINIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def simple_ukrainian_stem(word: str) -> str:
    for suffix in UKRAINIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def top_words(tokens: list[str], limit: int = 10) -> list[tuple[str, int]]:
    return Counter(tokens).most_common(limit)


def save_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_counter_csv(path: Path, items: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        writer.writerows(items)


def save_top_words_svg(path: Path, title: str, items: list[tuple[str, int]]) -> None:
    width = 920
    row_height = 34
    left = 170
    top = 70
    max_bar_width = 620
    height = top + len(items) * row_height + 45
    max_count = max((count for _, count in items), default=1)
    rows = []
    for index, (word, count) in enumerate(items):
        y = top + index * row_height
        bar_width = int((count / max_count) * max_bar_width)
        rows.append(
            f'<text x="{left - 14}" y="{y + 21}" text-anchor="end" class="label">{html.escape(word)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="22" rx="3" class="bar"/>'
            f'<text x="{left + bar_width + 10}" y="{y + 17}" class="value">{count}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #1f2937; }}
    .label {{ font: 15px Arial, sans-serif; fill: #374151; }}
    .value {{ font: 14px Arial, sans-serif; fill: #4b5563; }}
    .bar {{ fill: #3b82f6; }}
    .axis {{ stroke: #d1d5db; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="36" y="38" class="title">{html.escape(title)}</text>
  <line x1="{left}" y1="{top - 12}" x2="{left}" y2="{height - 34}" class="axis"/>
  {chr(10).join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_pipeline_comparison_csv(path: Path, results: list[PipelineResult]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "available", "elapsed_ms", "token_count", "unique_count", "without_stop_count"])
        for result in results:
            writer.writerow([
                result.name,
                result.available,
                result.elapsed_ms,
                result.token_count,
                result.unique_count,
                result.without_stop_count,
            ])


def save_pipeline_comparison_svg(path: Path, results: list[PipelineResult]) -> None:
    available_results = [result for result in results if result.available]
    width = 1080
    height = 520
    margin_left = 190
    margin_top = 70
    chart_width = 760
    group_height = 62
    bar_height = 12
    max_count = max((result.token_count for result in available_results), default=1)
    colors = {
        "token_count": "#2563eb",
        "unique_count": "#16a34a",
        "without_stop_count": "#f97316",
    }
    rows = []
    for index, result in enumerate(available_results):
        y = margin_top + index * group_height
        rows.append(f'<text x="{margin_left - 18}" y="{y + 22}" text-anchor="end" class="label">{html.escape(result.name)}</text>')
        for offset, key in enumerate(("token_count", "unique_count", "without_stop_count")):
            value = getattr(result, key)
            bar_width = int((value / max_count) * chart_width)
            bar_y = y + offset * (bar_height + 5)
            rows.append(
                f'<rect x="{margin_left}" y="{bar_y}" width="{bar_width}" height="{bar_height}" rx="2" fill="{colors[key]}"/>'
                f'<text x="{margin_left + bar_width + 8}" y="{bar_y + 10}" class="value">{value}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #1f2937; }}
    .label {{ font: 14px Arial, sans-serif; fill: #374151; }}
    .value {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .legend {{ font: 13px Arial, sans-serif; fill: #374151; }}
    .axis {{ stroke: #d1d5db; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="36" y="38" class="title">Pipeline comparison</text>
  <rect x="680" y="22" width="14" height="10" fill="{colors['token_count']}"/><text x="700" y="32" class="legend">tokens</text>
  <rect x="770" y="22" width="14" height="10" fill="{colors['unique_count']}"/><text x="790" y="32" class="legend">unique</text>
  <rect x="860" y="22" width="14" height="10" fill="{colors['without_stop_count']}"/><text x="880" y="32" class="legend">without stop words</text>
  <line x1="{margin_left}" y1="{margin_top - 16}" x2="{margin_left}" y2="{height - 55}" class="axis"/>
  {chr(10).join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def run_stdlib_pipeline(name: str, text: str, tokenizer, note: str) -> PipelineResult:
    started = time.perf_counter()
    tokens = tokenizer(text)
    without_stop = remove_stop_words(tokens)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return PipelineResult(
        name=name,
        available=True,
        elapsed_ms=round(elapsed_ms, 3),
        token_count=len(tokens),
        unique_count=len(set(tokens)),
        without_stop_count=len(without_stop),
        top_10=top_words(without_stop),
        note=note,
    )


def run_optional_nltk(text: str) -> PipelineResult:
    started = time.perf_counter()
    try:
        from nltk.tokenize import wordpunct_tokenize
    except Exception as exc:
        return PipelineResult("NLTK wordpunct", False, 0, 0, 0, 0, [], f"Не запущено: {exc}")
    tokens = [token.lower() for token in wordpunct_tokenize(text) if re.search(r"[А-Яа-яІіЇїЄєҐґA-Za-z]", token)]
    without_stop = remove_stop_words(tokens)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return PipelineResult(
        "NLTK wordpunct",
        True,
        round(elapsed_ms, 3),
        len(tokens),
        len(set(tokens)),
        len(without_stop),
        top_words(without_stop),
        "Альтернативна бібліотечна токенізація NLTK.",
    )


def run_optional_spacy(text: str) -> PipelineResult:
    started = time.perf_counter()
    try:
        import spacy

        nlp = spacy.load("uk_core_news_sm")
    except Exception as exc:
        return PipelineResult("spaCy uk_core_news_sm", False, 0, 0, 0, 0, [], f"Не запущено: {exc}")
    doc = nlp(text[:900_000])
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    without_stop = [token for token in tokens if token not in UKRAINIAN_STOP_WORDS]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return PipelineResult(
        "spaCy uk_core_news_sm",
        True,
        round(elapsed_ms, 3),
        len(tokens),
        len(set(tokens)),
        len(without_stop),
        top_words(without_stop),
        "Модельна NLP-обробка: токени, стоп-слова, потенційні леми та POS-теги.",
    )


def run_optional_pymorphy(tokens: list[str]) -> tuple[list[str], str]:
    try:
        import pymorphy3

        morph = pymorphy3.MorphAnalyzer(lang="uk")
    except Exception as exc:
        return [], f"pymorphy3 не запущено: {exc}"
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas, "pymorphy3: словникова лематизація українських слів."


def write_report(
    reports_dir: Path,
    source_path: Path,
    raw_text: str,
    filtered_text: str,
    normalized_text: str,
    word_tokens: list[str],
    no_stop_tokens: list[str],
    lemmas: list[str],
    stems: list[str],
    pipeline_results: list[PipelineResult],
    pymorphy_note: str,
) -> None:
    result_rows = [
        "| Варіант | Доступний | Час, мс | Токенів | Унікальних | Без стоп-слів | Коментар |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in pipeline_results:
        result_rows.append(
            f"| {result.name} | {'так' if result.available else 'ні'} | {result.elapsed_ms} | "
            f"{result.token_count} | {result.unique_count} | {result.without_stop_count} | {result.note} |"
        )

    lengths = [len(token) for token in no_stop_tokens]
    avg_len = round(statistics.mean(lengths), 2) if lengths else 0
    report = f"""# Лабораторна робота 2. NLP-конвеєр

## Мета роботи

Метою роботи є розроблення NLP-конвеєра для попередньої обробки текстового простору, отриманого в межах ЛР1, та порівняння кількох підходів до токенізації й мовної нормалізації.

## Вхідні дані

- Джерело: `{source_path}`
- Обсяг сирого тексту: {len(raw_text)} символів
- Обсяг після фільтрації: {len(filtered_text)} символів
- Обсяг після нормалізації: {len(normalized_text)} символів

## Виконані пункти 1.1-1.8

1. Вхідний текст збережено у `01_input_text.txt`.
2. Фільтрацію виконано у `02_filtered_text.txt`: прибрано цифри, службові символи, зайві пропуски.
3. Нормалізацію виконано у `03_normalized_text.txt`: нижній регістр та уніфікація апострофа.
4. Токенізацію виконано кількома способами, зокрема трьома основними для оцінювання:
   - `04_tokens_regex_words.txt`
   - `04_tokens_whitespace.txt`
   - `04_tokens_translate_table.txt`
   - `04_tokens_sentences.txt`
5. Стоп-слова видалено у `05_without_stop_words.txt`.
6. Лематизацію збережено у `06_lemmas_simple.txt`.
7. Стемінг збережено у `07_stems_suffix_rules.txt`.
8. TOP-10 тематичних слів збережено у `reports/metrics/08_top10_words.csv` і `reports/metrics/08_top10_words.json`; додатково створено TOP-10 лем у `reports/metrics/08_top10_lemmas.csv` і `reports/metrics/08_top10_lemmas.json`.

## TOP-10 тематичних слів

| Слово | Частота |
|---|---:|
{chr(10).join(f"| {word} | {count} |" for word, count in top_words(no_stop_tokens))}

Візуалізація: `reports/figures/top10_words.svg`.

## TOP-10 після лематизації

| Лема | Частота |
|---|---:|
{chr(10).join(f"| {word} | {count} |" for word, count in top_words(lemmas))}

## Порівняння альтернатив

{chr(10).join(result_rows)}

## Лематизація і стемінг

- Базова лематизація: словник винятків + прості правила для українських закінчень.
- Додаткова лематизація: {pymorphy_note}
- Стемінг: суфіксні правила для швидкого приведення слів до основи.
- Середня довжина слова після видалення стоп-слів: {avg_len}

## Висновок

Регулярний токенізатор дав стабільний результат для українських новинних заголовків, бо прибирає пунктуацію та не змішує її зі словами. Простий `split` швидший, але менш гнучкий для текстів зі складнішою пунктуацією. Бібліотечні варіанти NLTK, spaCy та pymorphy3 корисні для прикладних NLP-задач, оскільки дають відтворювану токенізацію та словникову лематизацію. Для цієї роботи практично придатним є поєднання базового stdlib-конвеєра з бібліотечною перевіркою результатів.
"""
    (reports_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Повний NLP-конвеєр для лабораторної роботи 2.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Шлях до текстового файлу.")
    parser.add_argument("--processed-output", type=Path, default=PROCESSED_DIR, help="Папка для проміжних текстових результатів.")
    parser.add_argument("--reports-output", type=Path, default=REPORTS_DIR, help="Папка для звіту.")
    parser.add_argument("--figures-output", type=Path, default=FIGURES_DIR, help="Папка для візуалізацій.")
    parser.add_argument("--metrics-output", type=Path, default=METRICS_DIR, help="Папка для метрик, CSV та JSON.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    processed_dir = args.processed_output.resolve()
    reports_dir = args.reports_output.resolve()
    figures_dir = args.figures_output.resolve()
    metrics_dir = args.metrics_output.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_text(input_path)
    filtered_text = filter_noise(raw_text)
    normalized_text = normalize_text(raw_text)

    regex_tokens = regex_word_tokenize(normalized_text)
    whitespace_tokens = whitespace_tokenize(normalized_text)
    translate_tokens = translate_table_tokenize(normalized_text)
    sentence_tokens = sentence_tokenize(filtered_text)
    without_stop = remove_stop_words(regex_tokens)
    simple_lemmas = [simple_ukrainian_lemma(token) for token in without_stop]
    simple_stems = [simple_ukrainian_stem(token) for token in without_stop]
    pymorphy_lemmas, pymorphy_note = run_optional_pymorphy(without_stop)
    report_lemmas = pymorphy_lemmas if pymorphy_lemmas else simple_lemmas

    save_lines(processed_dir / "01_input_text.txt", [raw_text])
    save_lines(processed_dir / "02_filtered_text.txt", [filtered_text])
    save_lines(processed_dir / "03_normalized_text.txt", [normalized_text])
    save_lines(processed_dir / "04_tokens_regex_words.txt", regex_tokens)
    save_lines(processed_dir / "04_tokens_whitespace.txt", whitespace_tokens)
    save_lines(processed_dir / "04_tokens_translate_table.txt", translate_tokens)
    save_lines(processed_dir / "04_tokens_sentences.txt", sentence_tokens)
    save_lines(processed_dir / "05_without_stop_words.txt", without_stop)
    save_lines(processed_dir / "06_lemmas_simple.txt", simple_lemmas)
    save_lines(processed_dir / "07_stems_suffix_rules.txt", simple_stems)
    if pymorphy_lemmas:
        save_lines(processed_dir / "06_lemmas_pymorphy3.txt", pymorphy_lemmas)

    top_10 = top_words(without_stop)
    top_10_lemmas = top_words(report_lemmas)
    save_counter_csv(metrics_dir / "08_top10_words.csv", top_10)
    save_counter_csv(metrics_dir / "08_top10_lemmas.csv", top_10_lemmas)
    save_top_words_svg(figures_dir / "top10_words.svg", "TOP-10 thematic words", top_10)
    save_top_words_svg(figures_dir / "top10_lemmas.svg", "TOP-10 lemmas", top_10_lemmas)
    (metrics_dir / "08_top10_words.json").write_text(
        json.dumps(dict(top_10), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metrics_dir / "08_top10_lemmas.json").write_text(
        json.dumps(dict(top_10_lemmas), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pipeline_results = [
        run_stdlib_pipeline("stdlib regex", normalized_text, regex_word_tokenize, "Практичний базовий варіант без зовнішніх залежностей."),
        run_stdlib_pipeline("stdlib split", normalized_text, whitespace_tokenize, "Найпростіша токенізація через пробіли; швидка, але менш точна."),
        run_stdlib_pipeline("stdlib translate table", normalized_text, translate_table_tokenize, "Практичний варіант через таблицю заміни пунктуації та split."),
        PipelineResult(
            "stdlib sentence regex",
            True,
            0,
            len(sentence_tokens),
            len(set(sentence_tokens)),
            len(sentence_tokens),
            [],
            "Токенізація на рівні речень для аналізу структури тексту.",
        ),
        run_optional_nltk(normalized_text),
        run_optional_spacy(normalized_text),
    ]

    (metrics_dir / "09_pipeline_comparison.json").write_text(
        json.dumps([asdict(result) for result in pipeline_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_pipeline_comparison_csv(metrics_dir / "09_pipeline_comparison.csv", pipeline_results)
    save_pipeline_comparison_svg(figures_dir / "pipeline_comparison.svg", pipeline_results)

    write_report(
        reports_dir,
        input_path,
        raw_text,
        filtered_text,
        normalized_text,
        regex_tokens,
        without_stop,
        report_lemmas,
        simple_stems,
        pipeline_results,
        pymorphy_note,
    )

    print(f"Готово. Проміжні дані збережено в: {processed_dir}")
    print(f"Звіт збережено в: {reports_dir}")
    print(f"Фігури збережено в: {figures_dir}")
    print(f"Метрики збережено в: {metrics_dir}")
    print("TOP-10 слів:")
    for word, count in top_10:
        print(f"{word}: {count}")


if __name__ == "__main__":
    main()
