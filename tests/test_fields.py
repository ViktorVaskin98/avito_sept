import pandas as pd

from avito_cg.data.params import ParamsParser
from avito_cg.index.fields import encoder_texts, item_fields, params_text, query_texts

PARSER = ParamsParser(
    ["Вид услуги", "Тип услуги", "Место оказания услуг", "Опыт работы", "График работы от"]
)

RAW = (
    "Вид услуги Автосервис, аренда Тип услуги Аренда авто "
    "Место оказания услуг Санкт-Петербург, Невский пр-т, 190 "
    "Опыт работы 10 График работы от 32400"
)


def test_keys_and_numbers_are_dropped():
    """В индекс должны попасть значения, а не служебные ключи и не время в секундах"""
    text = params_text(PARSER, [RAW])[0]
    assert "Автосервис, аренда" in text
    assert "Аренда авто" in text
    assert "Вид услуги" not in text
    assert "График работы" not in text
    assert "32400" not in text
    assert "10" not in text


def test_address_can_be_excluded():
    with_address = params_text(PARSER, [RAW], include_address=True)[0]
    without = params_text(PARSER, [RAW], include_address=False)[0]
    assert "Невский" in with_address
    assert "Невский" not in without
    assert "Аренда авто" in without


def test_item_fields_truncate_description():
    items = pd.DataFrame(
        {
            "item_title_raw": ["Тонировка авто"],
            "item_infm_params_text": [RAW],
            "item_description_raw": ["первое слово " * 200],
        }
    )
    fields = item_fields(items, PARSER, description_chars=50)
    assert set(fields) == {"title", "params", "description"}
    assert len(fields["description"][0]) <= 50
    assert len(item_fields(items, PARSER)["description"][0]) > 50


def test_query_text_optionally_carries_the_filter():
    queries = pd.DataFrame(
        {
            "search_query": ["кран"],
            "search_infm_params_text": ["Вид услуги Автосервис"],
        }
    )
    assert query_texts(queries) == ["кран"]
    assert query_texts(queries, with_filter=True) == ["кран Вид услуги Автосервис"]


def test_missing_text_does_not_crash_the_pipeline():
    """Пустое описание приезжает из parquet как NA и переживает astype(str)

    Наступил на это при сборке ответа по настоящему корпусу: в токенизатор прилетел
    float('nan'), который вдобавок истинный, так что проверка «if not text» его пропустила
    """
    items = pd.DataFrame(
        {
            "item_title_raw": pd.array(["Маникюр", None], dtype="string"),
            "item_infm_params_text": pd.array([RAW, None], dtype="string"),
            "item_description_raw": pd.array([None, "описание"], dtype="string"),
        }
    )
    fields = item_fields(items, PARSER)
    assert fields["description"][0] == ""
    assert fields["title"][1] == ""
    assert fields["params"][1] == ""


def test_normalize_survives_non_text():
    from avito_cg.data.text import normalize, tokenize

    assert normalize(float("nan")) == ""
    assert normalize(None) == ""
    assert tokenize(float("nan")) == []


def test_time_values_are_dropped():
    """«09:30» не ловится проверкой на цифры, а в параметрах его много"""
    from avito_cg.index.fields import params_values

    parser = ParamsParser(["Время работы, с", "Вид услуги"])
    values = params_values(parser, ["Время работы, с 09:30 Вид услуги Ремонт"])[0]
    assert "09:30" not in values
    assert "Ремонт" in values


def test_frequent_values_are_detected_by_share():
    """Отбираю мусорные значения по доле объявлений, а не чёрным списком"""
    from avito_cg.index.fields import frequent_values

    values = [
        ["Начальная цена", "Сантехника"],
        ["Начальная цена", "Электрика"],
        ["Начальная цена", "Кровля"],
        ["Начальная цена", "Плитка"],
    ]
    common = frequent_values(values, max_share=0.5)
    assert common == {"Начальная цена"}


def test_encoder_text_drops_common_values_but_keeps_rare_ones():
    parser = ParamsParser(["Вид услуги", "Тип стоимости за услугу"])
    items = pd.DataFrame(
        {
            "item_title_raw": ["Маникюр", "Электрика"],
            "item_infm_params_text": [
                "Вид услуги Красота Тип стоимости за услугу Начальная цена",
                "Вид услуги Ремонт Тип стоимости за услугу Начальная цена",
            ],
            "item_description_raw": ["", ""],
        }
    )
    texts = encoder_texts(items, parser, max_value_share=0.5)
    assert "Начальная цена" not in texts[0]
    assert "Красота" in texts[0]
    assert "Ремонт" in texts[1]
