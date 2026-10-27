# -*- coding: utf-8 -*-
"""
Учебная программа: один полностью прозрачный шаг обратного распространения.

Сеть 2 → 2 → 1, активация — сигмоида, потери L = 1/2(ŷ − y)².

Приложение показывает не результат, а путь к нему: формулу, подстановку и
число для каждого нейрона, каждый множитель цепного правила отдельно, градиент
каждого параметра и обновление w_new = w − η · ∂L/∂w.

Никаких сторонних библиотек: только tkinter и math. Соседние учебные проекты
серии не импортируются — файл самодостаточный.

Где что искать:
  1.  Пределы и палитра ............ LIMITS, COL_*
  2.  Форматирование чисел ......... f4(), f2(), signed(), term()
  3.  Математика сети .............. sigmoid(), forward(), backward(), chain_of()
  4.  Примеры ...................... EXAMPLES
  5.  Надписи (ru/en) .............. TEXTS / T(), справка GUIDE_RU и GUIDE_EN
  6.  Шаги и текст расчёта ......... STEPS, STEP_BY_N, calc_text()
  7.  Поля ввода .................. class ParamField
  8.  Граф сети ................... class NetworkView (tkinter Canvas)
  9.  Окно ........................ class BackpropApp
"""

import math
import tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------------------------------------------------------
# 1. Пределы ввода и палитра
# ---------------------------------------------------------------
# Все пределы мягкие: цель — не дать уехать числам в бесконечность, а не
# ограничивать учебные эксперименты. η до 1.5 оставлена намеренно: при
# заведомо большой скорости обучения видно, что шаг получается слишком большим.
LIMITS = {
    "x1": (-3.0, 3.0),
    "x2": (-3.0, 3.0),
    "y": (0.0, 1.0),
    "eta": (0.0, 1.5),
}
for _k in ("w1", "w2", "w3", "w4", "w5", "w6", "b_h1", "b_h2", "b_o"):
    LIMITS[_k] = (-3.0, 3.0)

COL_TEXT = "#242424"
COL_BG = "#fbfbf7"                  # фон окна
COL_PAPER = "#fdfdf9"               # фон «бумажных» панелей
COL_FWD = "#3f6f96"                 # прямой ход
COL_BWD = "#b3261e"                 # обратный ход
COL_NODE = "#eef3f7"
COL_NODE_LINE = "#8d949b"
COL_ACTIVE = "#b07d10"              # янтарная подсветка текущей стадии
COL_ACTIVE_FILL = "#fdf1d8"
COL_DIM = "#9a9a93"
COL_FRAME = "#c8c8c0"
COL_POS = "#2f7d4f"
COL_NEG = "#b3261e"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"
COL_CHAIN = "#5b3a86"

# ---------------------------------------------------------------
# 2. Форматирование чисел
# ---------------------------------------------------------------
# Настоящий математический минус (U+2212): в формулах он обязан выглядеть
# одинаково со всех сторон и не спорить с ASCII-дефисом.
MINUS = "−"


def f4(v):
    """Число для формул: 0.5 -> «0.5000», шум float -> «0.0000»."""
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.4f}".replace("-", MINUS)


def f2(v):
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.2f}".replace("-", MINUS)


def fmt(v):
    """Число для поля ввода: 2.0 -> «2», 0.25 -> «0.25».

    Здесь остаётся обычный ASCII-дефис: значение пользователь может напечатать
    в поле, и оно должно читаться обратно через float().
    """
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-10 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.6g}"


def signed(v):
    """Отрицательный множитель в произведении берём в скобки."""
    return f"({f4(v)})" if v < -5e-9 else f4(v)


def term(v):
    """Произведение ∂L/∂z · h: отрицательный член в скобках, знак — «·»."""
    return f"({f4(v)})" if v < -5e-9 else f4(v)


# ---------------------------------------------------------------
# 3. Математика сети: вся она в этом разделе, GUI только читает
# ---------------------------------------------------------------
# Обозначения — как в пояснениях к программе:
#   w1, w2 — веса входов x1, x2 у нейрона h1;  b_h1 — его смещение
#   w3, w4 — веса входов x1, x2 у нейрона h2;  b_h2 — его смещение
#   w5, w6 — веса h1, h2 у выходного нейрона;  b_o  — его смещение
HIDDEN_KEYS = ("w1", "w2", "b_h1", "w3", "w4", "b_h2")
OUTPUT_KEYS = ("w5", "w6", "b_o")
WEIGHT_KEYS = HIDDEN_KEYS + OUTPUT_KEYS
DATA_KEYS = ("x1", "x2", "y")
ALL_KEYS = DATA_KEYS + ("eta",) + WEIGHT_KEYS

DISP = {
    "x1": "x₁", "x2": "x₂", "y": "y", "eta": "η",
    "w1": "w₁", "w2": "w₂", "b_h1": "b_h₁",
    "w3": "w₃", "w4": "w₄", "b_h2": "b_h₂",
    "w5": "w₅", "w6": "w₆", "b_o": "b_o",
}
# Для подписи поля ввода: откуда идёт этот параметр. Подсказки нарочито
# математические — они не переводятся, как и сами обозначения.
EDGE_HINT = {
    "x1": "→ h₁ h₂", "x2": "→ h₁ h₂", "y": "→ L", "eta": "× ∂L/∂w",
    "w1": "x₁ → h₁", "w2": "x₂ → h₁", "b_h1": "h₁",
    "w3": "x₁ → h₂", "w4": "x₂ → h₂", "b_h2": "h₂",
    "w5": "h₁ → ŷ", "w6": "h₂ → ŷ", "b_o": "ŷ",
}


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(−z)), устойчиво и при больших |z|."""
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def forward(s):
    """Прямой ход: свёртки и активации. Возвращает z_h1, h1, z_h2, h2, z_o, ŷ."""
    f = {}
    f["z_h1"] = s["x1"] * s["w1"] + s["x2"] * s["w2"] + s["b_h1"]
    f["h1"] = sigmoid(f["z_h1"])
    f["z_h2"] = s["x1"] * s["w3"] + s["x2"] * s["w4"] + s["b_h2"]
    f["h2"] = sigmoid(f["z_h2"])
    f["z_o"] = f["h1"] * s["w5"] + f["h2"] * s["w6"] + s["b_o"]
    f["yhat"] = sigmoid(f["z_o"])
    return f


def loss_of(f, y):
    """L = 1/2 (ŷ − y)²."""
    return 0.5 * (f["yhat"] - y) ** 2


def backward(s, f):
    """Обратный ход по цепному правилу: каждый множитель лежит отдельно."""
    b = {}
    # Выход: ∂L/∂ŷ · ∂ŷ/∂z_o
    b["dL_dyhat"] = f["yhat"] - s["y"]
    b["dsig_o"] = f["yhat"] * (1.0 - f["yhat"])
    b["dL_dzo"] = b["dL_dyhat"] * b["dsig_o"]
    # Градиенты выходного слоя: ∂z_o/∂w = активация того входа, что идёт в w
    b["dL_dw5"] = b["dL_dzo"] * f["h1"]
    b["dL_dw6"] = b["dL_dzo"] * f["h2"]
    b["dL_dbo"] = b["dL_dzo"] * 1.0
    # Перенос ошибки в скрытый слой: ∂z_o/∂h = вес этого нейрона
    b["dL_dh1"] = b["dL_dzo"] * s["w5"]
    b["dL_dh2"] = b["dL_dzo"] * s["w6"]
    # Активация скрытого слоя: ∂h/∂z = h(1 − h)
    b["dsig_h1"] = f["h1"] * (1.0 - f["h1"])
    b["dsig_h2"] = f["h2"] * (1.0 - f["h2"])
    b["dL_dzh1"] = b["dL_dh1"] * b["dsig_h1"]
    b["dL_dzh2"] = b["dL_dh2"] * b["dsig_h2"]
    # Градиенты скрытого слоя: ∂z_h1/∂w1 = x1, ∂z_h1/∂b_h1 = 1
    b["dL_dw1"] = b["dL_dzh1"] * s["x1"]
    b["dL_dw2"] = b["dL_dzh1"] * s["x2"]
    b["dL_dbh1"] = b["dL_dzh1"] * 1.0
    b["dL_dw3"] = b["dL_dzh2"] * s["x1"]
    b["dL_dw4"] = b["dL_dzh2"] * s["x2"]
    b["dL_dbh2"] = b["dL_dzh2"] * 1.0
    return b


# Параметр -> символ его градиента и ключ в словаре backward().
GRAD_SYM = {
    "w1": ("∂L/∂w₁", "dL_dw1"), "w2": ("∂L/∂w₂", "dL_dw2"),
    "b_h1": ("∂L/∂b_h₁", "dL_dbh1"),
    "w3": ("∂L/∂w₃", "dL_dw3"), "w4": ("∂L/∂w₄", "dL_dw4"),
    "b_h2": ("∂L/∂b_h₂", "dL_dbh2"),
    "w5": ("∂L/∂w₅", "dL_dw5"), "w6": ("∂L/∂w₆", "dL_dw6"),
    "b_o": ("∂L/∂b_o", "dL_dbo"),
}


def grad_of(b, key):
    """Градиент одного параметра."""
    return b[GRAD_SYM[key][1]]


def chain_of(s, f, b, key):
    """Цепочка локальных множителей для одного параметра — по порядку цепного правила.

    Каждый элемент: символ производной, выражение буквами, подстановка числами,
    значение множителя и накопленное произведение `cum` (это уже настоящая
    производная потеря по данной величине). Именно из этих строк складываются и
    панель расчёта, и визуализация цепочки, поэтому разойтись они не могут.

    Важно: множители — только локальные производные (∂ одного шага графа). Если
    взять вместо ∂z_o/∂h₁ величину ∂L/∂h₁, она уже содержит ∂L/∂z_o, и она же
    стоит первым множителем — произведение дважды посчитало бы один и тот же
    множитель, а градиент вышел бы неверным.
    """
    out = [
        {"sym": "∂L/∂ŷ", "expr": "ŷ − y",
         "sub": f"{f4(f['yhat'])} {MINUS} {f4(s['y'])}", "val": b["dL_dyhat"],
         "cum_label": "∂L/∂ŷ"},
        {"sym": "∂ŷ/∂z_o", "expr": "ŷ(1 − ŷ)",
         "sub": f"{f4(f['yhat'])}·(1 {MINUS} {f4(f['yhat'])})", "val": b["dsig_o"],
         "cum_label": "∂L/∂z_o"},
    ]
    cum = 1.0
    for item in out:
        cum *= item["val"]
        item["cum"] = cum

    if key in ("w5", "w6", "b_o"):
        last = {
            "w5": ("∂z_o/∂w₅", "h₁", f4(f["h1"]), f["h1"]),
            "w6": ("∂z_o/∂w₆", "h₂", f4(f["h2"]), f["h2"]),
            "b_o": ("∂z_o/∂b_o", "1", "1.0000", 1.0),
        }[key]
    else:
        hnum = "1" if key in ("w1", "w2", "b_h1") else "2"
        h = f["h" + hnum]
        wkey = "w5" if hnum == "1" else "w6"
        out.append({"sym": f"∂z_o/∂h{hnum}", "expr": wkey.replace("_", ""),
                    "sub": f4(s[wkey]), "val": s[wkey],
                    "cum_label": f"∂L/∂h{hnum}"})
        out.append({"sym": f"∂h{hnum}/∂z_h{hnum}", "expr": f"h{hnum}(1 − h{hnum})",
                    "sub": f"{f4(h)}·(1 {MINUS} {f4(h)})",
                    "val": b["dsig_h" + hnum],
                    "cum_label": f"∂L/∂z_h{hnum}"})
        last = {
            "w1": ("∂z_h₁/∂w₁", "x₁", f4(s["x1"]), s["x1"]),
            "w2": ("∂z_h₁/∂w₂", "x₂", f4(s["x2"]), s["x2"]),
            "b_h1": ("∂z_h₁/∂b_h₁", "1", "1.0000", 1.0),
            "w3": ("∂z_h₂/∂w₃", "x₁", f4(s["x1"]), s["x1"]),
            "w4": ("∂z_h₂/∂w₄", "x₂", f4(s["x2"]), s["x2"]),
            "b_h2": ("∂z_h₂/∂b_h₂", "1", "1.0000", 1.0),
        }[key]

    out.append({"sym": last[0], "expr": last[1], "sub": last[2], "val": last[3],
                "cum_label": GRAD_SYM[key][0]})
    # Накопленное произведение: после каждого множителя — своя производная
    # потеря. Последнее накопленное произведение и есть градиент параметра.
    cum = 1.0
    for item in out:
        cum *= item["val"]
        item["cum"] = cum
    return out


def product_str(chain):
    """Строка произведения всех множителей цепочки."""
    return " · ".join(term(c["val"]) for c in chain)


def updated_state(s, b):
    """Параметры после шага градиентного спуска: w_new = w − η · ∂L/∂w."""
    new = dict(s)
    for k in WEIGHT_KEYS:
        new[k] = s[k] - s["eta"] * grad_of(b, k)
    return new


def delta_of(s, b, key):
    """Δw = −η · ∂L/∂w."""
    return -s["eta"] * grad_of(b, key)


# ---------------------------------------------------------------
# 4. Примеры
# ---------------------------------------------------------------
# Каждый пример — полностью валидные числа; градиенты каждого параметра сверены
# с конечными разностями потерь и с ручным счётом по формулам выше.
def example_state(x1, x2, y, eta, w1, w2, bh1, w3, w4, bh2, w5, w6, bo):
    return {"x1": x1, "x2": x2, "y": y, "eta": eta,
            "w1": w1, "w2": w2, "b_h1": bh1,
            "w3": w3, "w4": w4, "b_h2": bh2,
            "w5": w5, "w6": w6, "b_o": bo}


EXAMPLES = (
    {   # 1. Основной пример: числа читаются вручную, ошибка заметна.
        "id": "ex1",
        "state": example_state(1.0, 0.8, 1.0, 0.30,
                               0.5, -0.2, 0.1, 0.3, 0.4, -0.1, 0.7, -0.3, 0.1),
    },
    {   # 2. Цель 0: тот же прогноз, но ошибка противоположного знака.
        "id": "ex2",
        "state": example_state(1.0, 0.8, 0.0, 0.30,
                               0.5, -0.2, 0.1, 0.3, 0.4, -0.1, 0.7, -0.3, 0.1),
    },
    {   # 3. Прогноз близок к цели: малые ошибка, потери и градиенты.
        "id": "ex3",
        "state": example_state(1.0, 0.8, 1.0, 0.30,
                               0.5, -0.2, 0.1, 0.3, 0.4, -0.1, 0.7, -0.3, 1.2),
    },
    {   # 4. Та же сеть, но η заведомо мала: обновление едва заметно.
        "id": "ex4",
        "state": example_state(1.0, 0.8, 1.0, 0.02,
                               0.5, -0.2, 0.1, 0.3, 0.4, -0.1, 0.7, -0.3, 0.1),
    },
    {   # 5. Отрицательный вход и отрицательные веса: знаки градиентов чередуются.
        "id": "ex5",
        "state": example_state(1.5, -1.0, 1.0, 0.30,
                               -0.6, 0.8, 0.2, 0.4, -0.5, 0.3, 0.9, 0.6, -0.4),
    },
    {   # 6. Насыщение сигмоиды: σ'(z) ≈ 0, поэтому градиенты почти нулевые.
        "id": "ex6",
        "state": example_state(2.5, 2.5, 1.0, 0.30,
                               2.0, 2.0, 1.0, 1.5, 1.5, 1.0, 2.0, 2.0, 1.0),
    },
)
EXAMPLE_IDS = [ex["id"] for ex in EXAMPLES]
DEFAULT_EXAMPLE = EXAMPLES[0]

# ---------------------------------------------------------------
# 5. Формулы: они одинаковы на обоих языках, поэтому не переводятся
# ---------------------------------------------------------------
FORMULA_SIGMOID = "σ(z) = 1 / (1 + e^(−z))"
FORMULA_DSIGMOID = "σ′(z) = σ(z)·(1 − σ(z))"
FORMULA_LOSS = "L = 1/2·(ŷ − y)²"
FORMULA_UPDATE = "w_new = w − η · ∂L/∂w"
FORMULA_Z_H1 = "z_h₁ = x₁·w₁ + x₂·w₂ + b_h₁"
FORMULA_Z_H2 = "z_h₂ = x₁·w₃ + x₂·w₄ + b_h₂"
FORMULA_Z_O = "z_o = h₁·w₅ + h₂·w₆ + b_o"

# ---------------------------------------------------------------
# 6. Надписи интерфейса: весь перевод живёт здесь (русский по умолчанию)
# ---------------------------------------------------------------
LANG = "ru"

GUIDE_RU = """1. ЧТО ТАКОЕ ОБРАТНОЕ РАСПРОСТРАНЕНИЕ
Обратное распространение ошибки (backpropagation) отвечает на один вопрос: если
я чуть-чуть подвину вот этот вес, насколько изменятся потери? Ответ на этот
вопрос — производная потерь по весу, то есть градиент. Зная градиенты всех
весов и смещений, мы можем подвинуть каждый параметр в нужную сторону.

Метод называется «обратным», потому что считает эти производные справа налево:
от потерь к предсказанию, от предсказания к скрытому слою, от скрытого слоя к
весам.

2. ПРЯМОЙ ХОД
Прямой ход — это обычное вычисление ответа сети: входы умножаются на веса
скрытого слоя, складываются со смещениями, прогоняются через активацию, затем
так же считается выход. Прямой ход нужен дважды: чтобы получить предсказание и
чтобы сохранить все промежуточные числа (h₁, h₂, ŷ), без которых градиенты не
вычислить.

3. ВЗВЕШЕННАЯ СУММА
z = x₁·w₁ + x₂·w₂ + b — это и есть свёртка: каждый вход входит в неё со своим
весом, а смещение b добавляется отдельно. Смысл веса — «насколько сильно этот
вход влияет на нейрон», смысл смещения — «насколько сдвинут порог реакции».

4. АКТИВАЦИЯ
Сигмоида σ(z) = 1 / (1 + e^(−z)) сжимает любое число в промежуток от 0 до 1.
Она нужна, чтобы сеть была нелинейной: без функции активации два слоя свёрток
сложились бы в одну свёртку, и скрытый слой стал бы бесполезен.
Производная сигмоиды записывается через её собственное значение:
σ′(z) = σ(z)·(1 − σ(z)). Поэтому, когда мы уже посчитали h = σ(z), производная
бесплатна: h·(1 − h).

5. ФУНКЦИЯ ПОТЕРЬ
L = 1/2·(ŷ − y)² отвечает на вопрос «насколько сеть ошиблась». Множитель 1/2
введён только ради красоты производной: ∂L/∂ŷ = ŷ − y, двойка сокращается.
Потери — единственная величина, от которой мы отталкиваемся в обратном ходе:
именно к ней цепляются все производные.

6. ЦЕПНОЕ ПРАВИЛО
Если одна величина зависит от другой, а та — от третьей, то производная
«дальней» величины по «ближней» равна произведению локальных производных:

    ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z_o · ∂z_o/∂w

Каждый множитель здесь — маленькая, легко считаемая производная одного шага
графа. Цепное правило — это единственный способ посчитать градиенты глубоко в
сети, не перебирая варианты числами.

Важный нюанс, который легко испортить в расчёте: множить надо локальные
производные (∂z_o/∂h₁ = w₅), а не уже готовые производные потерь (∂L/∂h₁).
Величина ∂L/∂h₁ сама содержит ∂L/∂z_o, и если поставить её множителем рядом с
∂L/∂z_o, один и тот же множитель войдёт в произведение дважды.

7. ГРАДИЕНТЫ
Градиент параметра показывает, куда и как сильно изменятся L, если параметр
увеличить. Отрицательный градиент — значит «увеличивать параметр полезно,
потери падают». Чем больше модуль градиента, тем чувствительнее потери к этому
параметру. У разных весов градиенты разные, потому что у каждого веса свой путь
ошибки к потерям: свои входы, своя активация, свой множитель сигмоиды.

8. ОБРАТНЫЙ ХОД
Обратный ход — это цепное правило, применённое один раз для всего графа и
использованное повторно. Сначала считается ∂L/∂z_o (это и есть «ошибка
выходного нейрона»), затем она переносится в скрытый слой через веса w₅ и w₆,
умножается на производные активаций скрытого слоя — и только после этого
получаются градиенты весов w₁…w₄. Порядок принципиален: без ∂L/∂z_o нельзя
взять ∂L/∂w₁.

9. ГРАДИЕНТНЫЙ СПУСК
w_new = w − η · ∂L/∂w. Мы идём против направления градиента, потому что
градиент указывает, где потери растут. η (скорость обучения) — единственное,
что определяет размер шага: мала —параметры почти не двигаются, велика — шаг
проскакивает минимум и потери растут. Один шаг не обязан уменьшать потери: при
неподходящей η они могут и вырасти. Программа честно показывает, что вышло.

10. ОДИН ПОЛНЫЙ ШАГ ОБУЧЕНИЯ
Прямой ход → предсказание → потери → обратный ход → градиенты → обновление →
пересчёт. Это и есть один шаг обучения, ровно тот, который в реальных системах
повторяют тысячи раз. Здесь он сделан один раз и показан целиком: каждое число
можно проверить руками до того, как оно появится на экране.

Автор: SteveWorks001
"""

GUIDE_EN = """1. WHAT BACKPROPAGATION IS
Backpropagation answers a single question: if I nudge this weight a little, how
much does the loss change? That answer is the derivative of the loss with
respect to the weight — the gradient. Once we know the gradients of all weights
and biases, we can move every parameter in the right direction.

It is called "backward" because it computes these derivatives from right to
left: from the loss to the prediction, from the prediction to the hidden layer,
from the hidden layer to the weights.

2. THE FORWARD PASS
The forward pass is the ordinary computation of the network's answer: inputs are
multiplied by the hidden-layer weights, biases are added, activations are
applied, and the output is computed the same way. We need it twice: to get a
prediction, and to keep every intermediate value (h1, h2, y-hat) without which
the gradients cannot be computed.

3. THE WEIGHTED SUM
z = x1*w1 + x2*w2 + b is the weighted sum: every input enters it with its own
weight, and the bias b is added separately. A weight means "how strongly this
input drives the neuron"; a bias means "how far the neuron's reaction threshold
is shifted".

4. ACTIVATION
The sigmoid σ(z) = 1 / (1 + e^(−z)) compresses any number into the range 0 to 1.
It makes the network nonlinear: without an activation function two layers of
weighted sums would collapse into a single one and the hidden layer would be
useless. The derivative of the sigmoid is written through its own value:
σ′(z) = σ(z)·(1 − σ(z)). So once h = σ(z) is known, the derivative is free:
h·(1 − h).

5. THE LOSS FUNCTION
L = 1/2·(ŷ − y)² answers "how wrong was the network". The 1/2 factor exists only
to keep the derivative pretty: ∂L/∂ŷ = ŷ − y, the two cancels. The loss is the
one quantity everything hangs on in the backward pass: every derivative is
measured against it.

6. THE CHAIN RULE
If one quantity depends on another, and that one on a third, the derivative of
the "far" quantity through the "near" one is the product of local derivatives:

    ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z_o · ∂z_o/∂w

Each factor is a small, easy derivative of one step of the graph. The chain rule
is the only way to compute gradients deep inside a network without probing them
numerically one by one.

One subtlety that is easy to get wrong: multiply LOCAL derivatives
(∂z_o/∂h₁ = w₅), not already-accumulated loss derivatives (∂L/∂h₁). The value
∂L/∂h₁ already contains ∂L/∂z_o, so using it as a factor next to ∂L/∂z_o makes
the same factor enter the product twice.

7. GRADIENTS
The gradient of a parameter says how the loss moves when that parameter grows.
A negative gradient means "growing this parameter lowers the loss". The larger
the magnitude, the more sensitive the loss is to that parameter. Different
weights have different gradients because each has its own path from the error to
the loss: its own input, its own activation, its own sigmoid factor.

8. THE BACKWARD PASS
The backward pass is the chain rule applied once to the whole graph and reused.
First ∂L/∂z_o (the "error of the output neuron"), then it is carried into the
hidden layer through the weights w₅ and w₆, multiplied by the hidden activation
derivatives — and only then come the gradients of w₁…w₄. The order matters:
without ∂L/∂z_o there is no ∂L/∂w₁.

9. GRADIENT DESCENT
w_new = w − η · ∂L/∂w. We move against the gradient because the gradient points
where the loss grows. η (the learning rate) is the only thing that sets the step
size: too small and the parameters barely move, too large and the step jumps
past the minimum while the loss grows. One step is not obliged to reduce the
loss: with a bad η it can increase it. The program reports what actually
happened.

10. ONE COMPLETE LEARNING STEP
Forward pass → prediction → loss → backward pass → gradients → update →
recomputation. That is one training step, exactly the one real systems repeat
thousands of times. Here it happens once and is shown in full: every number can
be checked by hand before it reaches the screen.

Author: SteveWorks001
"""


def _safe_format(template, **kw):
    """Подстановка {имён}; неизвестное оставляем как есть, чтобы не падать."""
    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_Safe(kw))


def T(key, **kw):
    """Надпись текущего языка; подстановки {name} — только если переданы."""
    text = TEXTS[LANG][key]
    return _safe_format(text, **kw) if kw else text


def set_language(code):
    """Сменить язык (перерисовку виджетов делает retranslate)."""
    global LANG
    LANG = code


def guide_text():
    """Справка на текущем языке (10 разделов)."""
    return TEXTS[LANG]["guide"]


TEXTS = {
    "ru": {
        "app_title": "Обратное распространение ошибки",
        "subtitle": "учебный симулятор · один полный шаг обучения сети 2 → 2 → 1",
        "lang_label": "Язык:",
        "lang_ru": "Русский",
        "lang_en": "English",

        "box_examples": "Примеры",
        "ex1_name": "1. Основной пример",
        "ex1_desc": "ошибка заметна, числа читаются вручную",
        "ex2_name": "2. Цель равна 0",
        "ex2_desc": "та же сеть, но знак ошибки противоположный",
        "ex3_name": "3. Прогноз близок к цели",
        "ex3_desc": "малые потери — малые градиенты",
        "ex4_name": "4. Очень малая η",
        "ex4_desc": "параметры сдвигаются едва заметно",
        "ex5_name": "5. Отрицательные веса и вход",
        "ex5_desc": "знаки градиентов чередуются",
        "ex6_name": "6. Насыщение сигмоиды",
        "ex6_desc": "σ′(z) ≈ 0, поэтому градиенты почти нулевые",

        "box_mode": "Режим",
        "mode_fwd": "Прямой ход",
        "mode_bwd": "Обратный ход",
        "mode_full": "Полный шаг обучения",
        "mode_fwd_hint": "Только вычисление ответа сети: входы → скрытый слой → выход → потери.",
        "mode_bwd_hint": "Только обратный ход: ошибка переносится от потерь к весам цепным правилом.",
        "mode_full_hint": "Вся цепочка целиком: прямой ход, потери, градиенты, обновление, пересчёт.",

        "box_data": "Данные и скорость обучения",
        "box_hidden": "Скрытый слой (2 нейрона)",
        "box_output": "Выходной нейрон",
        "box_progress": "Движение по шагам",
        "input_hint": "Можно печатать точку и запятую: 0,5 и 0.5 одинаковы.",

        "btn_next": "Следующий шаг",
        "btn_prev": "Предыдущий шаг",
        "btn_reset": "Сброс",
        "btn_full": "Показать весь процесс",
        "lbl_step": "Шаг {n} из {total}",
        "lbl_step_name": "{title}",
        "lbl_phase_fwd": "прямой ход",
        "lbl_phase_bwd": "обратный ход",
        "lbl_phase_upd": "обновление",

        "st1_t": "Входы, цель и параметры сети",
        "st2_t": "Свёртка нейрона h₁",
        "st3_t": "Активация нейрона h₁",
        "st4_t": "Свёртка нейрона h₂",
        "st5_t": "Активация нейрона h₂",
        "st6_t": "Свёртка выходного нейрона",
        "st7_t": "Предсказание и потери",
        "st8_t": "Градиент выходного слоя",
        "st9_t": "Градиенты скрытого слоя",
        "st10_t": "Обновление всех параметров",
        "st11_t": "Пересчёт: до и после шага",

        "why1": "Всё начинается с этих чисел. {x1} и {x2} идут в скрытый слой, "
                "y — это ответ, который мы хотим получить, а η определяет размер "
                "будущего шага. Обратите внимание на знаки: вес со знаком минус "
                "тоже участвует в расчёте наравне с остальными.",
        "why2": "Свёртка — это единственная линейная часть нейрона: входы "
                "умножаются на свои веса, смещение добавляется отдельно. Здесь "
                "z_h₁ = {z1}. Пока это число ничем не отличается от z_h₂ — обе "
                "свёртки считаются одинаково.",
        "why3": "Сигмоида сжимает свёртку в промежуток от 0 до 1: h₁ = {h1}. "
                "Именно это значение потом пойдёт в выходной нейрон, и именно оно "
                "понадобится в обратном ходе как множитель ∂z_o/∂h₁. Запомните "
                "его: обратный ход не пересчитывает прямой, а использует его "
                "числа.",
        "why4": "Тот же расчёт для второго нейрона скрытого слоя: z_h₂ = {z2}. "
                "Два нейрона считаются независимо — между ними связей нет.",
        "why5": "h₂ = {h2}. Теперь оба числа скрытого слоя известны, и из них "
                "можно собрать ответ сети.",
        "why6": "Выходной нейрон устроен так же, но его входами служат активации "
                "скрытого слоя, а не исходные x: z_o = {zo}. Здесь работают веса "
                "w₅ и w₆ — те самые, чьи градиенты мы потом получим первыми.",
        "why7": "ŷ = {yhat} против цели y = {y}. Ошибка ŷ − y = {err}, а потери "
                "L = {loss}. На этом прямой ход закончен: всё, что нужно для "
                "обратного, уже посчитано и сохранено.",
        "why8": "Обратный ход начинается с единственного вопроса: как меняются "
                "потери, когда меняется ŷ. Ответ — ∂L/∂ŷ = {dL_dyhat}. Дальше "
                "∂ŷ/∂z_o = σ′(z_o) = {dsig_o}, и их произведение даёт "
                "∂L/∂z_o = {dL_dzo} — ошибку выходного нейрона. Умножив её на "
                "входы этого нейрона, получаем градиенты w₅, w₆ и b_o.",
        "why9": "Чтобы дойти до весов скрытого слоя, ошибку надо перенести через "
                "сами веса: ∂L/∂h₁ = ∂L/∂z_o · w₅ = {dL_dh1}, затем "
                "∂L/∂z_h₁ = ∂L/∂h₁ · σ′(z_h₁) = {dL_dzh1}. Обратите внимание: "
                "множим именно на локальные производные. Если поставить вместо "
                "w₅ готовое ∂L/∂h₁, множитель ∂L/∂z_o войдёт в произведение "
                "дважды, и градиент станет неверным.",
        "why10": "Каждый параметр сдвигается на −η · ∂L/∂параметр. Где градиент "
                 "отрицательный, параметр растёт; где положительный — уменьшается; "
                 "где градиент равен нулю — остаётся на месте. Скорость обучения η = "
                 "{eta} одна и та же для всех: её множитель общий, анаправления разные.",
        "why11": "Те же самые вычисления, но с обновлёнными параметрами: "
                 "ŷ было {yhat0}, стало {yhat1}; потери были {l0}, стали {l1}. "
                 "Это и есть один шаг обучения. Он не обязан уменьшать потери при "
                 "любом η — здесь потери {dir_word}.",

        "tab_calc": "Расчёт",
        "tab_grads": "Градиенты",
        "tab_why": "Почему?",
        "tab_guide": "Справка",

        "th_param": "Параметр",
        "th_value": "Значение",
        "th_grad": "∂L/∂w",
        "th_delta": "Изменение",
        "th_new": "Новое",

        "hdr_before": "ДО ОБНОВЛЕНИЯ",
        "hdr_after": "ПОСЛЕ ОБНОВЛЕНИЯ",
        "c_inputs": "Входы и цель",
        "c_params": "Параметры сети",
        "c_forward": "ПРЯМОЙ ХОД",
        "c_backward": "ОБРАТНЫЙ ХОД",
        "c_update": "ОБНОВЛЕНИЕ ПАРАМЕТРОВ",
        "c_recalc": "ПЕРЕСЧЁТ ПОСЛЕ ШАГА",
        "c_hidden1": "Нейрон h₁",
        "c_hidden2": "Нейрон h₂",
        "c_output": "Выходной нейрон",
        "c_pred": "Предсказание",
        "c_target": "Цель",
        "c_err": "Ошибка ŷ − y",
        "c_loss": "Потери",
        "c_chain": "Цепное правило по шагам",
        "c_grad_out": "Градиенты выходного слоя",
        "c_grad_hidden": "Градиенты скрытого слоя",
        "c_sigma": "Сигмоида и её производная",
        "c_result": "Итог",
        "c_loss_before": "Потери до шага",
        "c_loss_after": "Потери после шага",
        "c_pred_before": "Предсказание до шага",
        "c_pred_after": "Предсказание после шага",
        "c_notyet": "Этот шаг ещё не выполнен — нажмите «Следующий шаг».",
        "c_stepdone": "Выполнено шагов: {n} из {total}",
        "c_rate": "Скорость обучения",
        "dir_down": "уменьшились",
        "dir_up": "выросли",
        "dir_same": "остались практически теми же",

        "what_hdr": "Что произошло?",
        "what_down": "Потери уменьшились с {l0} до {l1} (на {d}): шаг выбран в "
                     "сторону убывания потерь, и на этом направлении они действительно упали.",
        "what_up": "Потери выросли с {l0} до {l1} (на {d}): направление шага верное, "
                   "но сам шаг слишком велик — он проскочил окрестность минимума. "
                   "Уменьшите η.",
        "what_same": "Потери практически не изменились ({l0} → {l1}): градиенты "
                     "малы, поэтому и сдвиг параметров мал.",
        "what_neg": "У параметра {k} градиент отрицательный ({g}), поэтому вычитание "
                    "−η·∂L/∂w увеличило его на {d}.",
        "what_pos": "У параметра {k} градиент положительный ({g}), поэтому он "
                    "уменьшился на {d}.",
        "what_zero": "У параметра {k} градиент равен {g}: этот параметр на текущем "
                     "шаге не двигается.",
        "what_worst": "Сильнее всего изменился параметр {k}: у него наибольший по "
                      "модулю градиент ({g}).",
        "what_sat": "Сигмоида насыщена (ŷ = {yhat}), её производная σ′(z_o) = "
                    "{dsig_o} близка к нулю — вот почему все градиенты такие малые.",

        "legend_fwd": "Прямой ход: слева → направо",
        "legend_bwd": "Обратный ход: справа → налево",
        "chain_hdr": "Цепочка зависимостей для параметра {k}",
        "chain_none": "Цепочка пока не выбрана",
        "chain_hint": "Кликните строку в таблице «Градиенты», чтобы увидеть цепочку "
                      "множителей для этого параметра.",

        "status_ready": "Готово. Нажмите «Следующий шаг» — он выполнит ровно одну стадию.",
        "status_lang": "Язык интерфейса переключён.",
        "status_step": "Шаг {n}: {title}",
        "status_first": "Это первый шаг: возвращаться некуда.",
        "status_last": "Достигнут конец процесса: шаг {n} — последний в этом режиме.",
        "status_reset": "Сброс: прогресс очищен, параметры остались вашими.",
        "status_example": "Загружен пример «{name}»: все поля обновлены.",
        "status_full": "Показан весь процесс целиком: {n} стадий.",
        "status_bad": "Не число: «{raw}». Оставлено последнее правильное значение.",
        "status_range": "{name}: допустимо от {lo} до {hi}. Взято ближайшее разрешённое.",
        "status_mode": "Режим: {mode}. Доступны шаги {a}–{b}.",
        "status_selected": "Цепочка показана для параметра {k}.",

        "guide": GUIDE_RU,
    },

    "en": {
        "app_title": "Backpropagation",
        "subtitle": "an educational simulator · one complete learning step of a 2 → 2 → 1 network",
        "lang_label": "Language:",
        "lang_ru": "Русский",
        "lang_en": "English",

        "box_examples": "Examples",
        "ex1_name": "1. Main example",
        "ex1_desc": "the error is visible, the numbers are readable by hand",
        "ex2_name": "2. Target equals 0",
        "ex2_desc": "same network, opposite sign of the error",
        "ex3_name": "3. Prediction close to target",
        "ex3_desc": "small loss — small gradients",
        "ex4_name": "4. Very small η",
        "ex4_desc": "the parameters barely move",
        "ex5_name": "5. Negative weights and input",
        "ex5_desc": "gradient signs alternate",
        "ex6_name": "6. Saturating sigmoid",
        "ex6_desc": "σ′(z) ≈ 0, so the gradients are almost zero",

        "box_mode": "Mode",
        "mode_fwd": "Forward pass",
        "mode_bwd": "Backward pass",
        "mode_full": "Full learning step",
        "mode_fwd_hint": "Only computing the network's answer: inputs → hidden layer → output → loss.",
        "mode_bwd_hint": "Only the backward pass: the error is carried from the loss to the weights.",
        "mode_full_hint": "The whole chain: forward pass, loss, gradients, update, recomputation.",

        "box_data": "Data and learning rate",
        "box_hidden": "Hidden layer (2 neurons)",
        "box_output": "Output neuron",
        "box_progress": "Step through",
        "input_hint": "Both separators work: 0.5 and 0,5 mean the same.",

        "btn_next": "Next step",
        "btn_prev": "Previous step",
        "btn_reset": "Reset",
        "btn_full": "Show full process",
        "lbl_step": "Step {n} of {total}",
        "lbl_step_name": "{title}",
        "lbl_phase_fwd": "forward pass",
        "lbl_phase_bwd": "backward pass",
        "lbl_phase_upd": "update",

        "st1_t": "Inputs, target and network parameters",
        "st2_t": "Weighted sum of neuron h₁",
        "st3_t": "Activation of neuron h₁",
        "st4_t": "Weighted sum of neuron h₂",
        "st5_t": "Activation of neuron h₂",
        "st6_t": "Weighted sum of the output neuron",
        "st7_t": "Prediction and loss",
        "st8_t": "Gradient of the output layer",
        "st9_t": "Gradients of the hidden layer",
        "st10_t": "Updating every parameter",
        "st11_t": "Recomputation: before and after",

        "why1": "Everything starts from these numbers. {x1} and {x2} feed the hidden "
                "layer, y is the answer we want, and η sets the size of the step to "
                "come. Note the signs: a negative weight takes part just like the rest.",
        "why2": "The weighted sum is the only linear part of a neuron: inputs are "
                "multiplied by their own weights and the bias is added separately. "
                "Here z_h₁ = {z1}. For now this number is no different from z_h₂ — "
                "both sums are computed the same way.",
        "why3": "The sigmoid compresses the sum into the range 0 to 1: h₁ = {h1}. "
                "This very value goes into the output neuron, and the backward pass "
                "will reuse it as the factor ∂z_o/∂h₁. Remember that: backpropagation "
                "does not recompute the forward pass, it uses the numbers it kept.",
        "why4": "The same computation for the second hidden neuron: z_h₂ = {z2}. The "
                "two neurons are computed independently — there are no links between them.",
        "why5": "h₂ = {h2}. Both hidden activations are known now, so the answer of "
                "the network can be assembled.",
        "why6": "The output neuron works the same way, but its inputs are the hidden "
                "activations rather than the original x: z_o = {zo}. These are the "
                "weights w₅ and w₆ whose gradients we get first.",
        "why7": "ŷ = {yhat} against the target y = {y}. The error ŷ − y = {err}, and "
                "the loss L = {loss}. The forward pass ends here: everything the "
                "backward pass needs is already computed and kept.",
        "why8": "The backward pass starts with a single question: how does the loss "
                "change when ŷ changes? The answer is ∂L/∂ŷ = {dL_dyhat}. Then "
                "∂ŷ/∂z_o = σ′(z_o) = {dsig_o}, and their product gives "
                "∂L/∂z_o = {dL_dzo} — the error of the output neuron. Multiplying it "
                "by that neuron's inputs gives the gradients of w₅, w₆ and b_o.",
        "why9": "To reach the hidden weights the error must be carried through the "
                "weights themselves: ∂L/∂h₁ = ∂L/∂z_o · w₅ = {dL_dh1}, then "
                "∂L/∂z_h₁ = ∂L/∂h₁ · σ′(z_h₁) = {dL_dzh1}. Note that we multiply by "
                "local derivatives only. Putting the ready-made ∂L/∂h₁ in place of w₅ "
                "would make ∂L/∂z_o enter the product twice and the gradient would be wrong.",
        "why10": "Every parameter moves by −η · ∂L/∂parameter. A negative gradient "
                 "grows the parameter, a positive one shrinks it, a zero gradient "
                 "leaves it in place. The learning rate η = {eta} is the same for all: "
                 "the multiplier is shared, the directions differ.",
        "why11": "The same computation with the updated parameters: ŷ was {yhat0} and "
                 "became {yhat1}; the loss was {l0} and became {l1}. This is one "
                 "learning step. It is not obliged to reduce the loss for every η — "
                 "here the loss {dir_word}.",

        "tab_calc": "Calculation",
        "tab_grads": "Gradients",
        "tab_why": "Why?",
        "tab_guide": "Guide",

        "th_param": "Parameter",
        "th_value": "Value",
        "th_grad": "∂L/∂w",
        "th_delta": "Change",
        "th_new": "New",

        "hdr_before": "BEFORE THE UPDATE",
        "hdr_after": "AFTER THE UPDATE",
        "c_inputs": "Inputs and target",
        "c_params": "Network parameters",
        "c_forward": "FORWARD PASS",
        "c_backward": "BACKWARD PASS",
        "c_update": "PARAMETER UPDATE",
        "c_recalc": "RECOMPUTATION AFTER THE STEP",
        "c_hidden1": "Neuron h₁",
        "c_hidden2": "Neuron h₂",
        "c_output": "Output neuron",
        "c_pred": "Prediction",
        "c_target": "Target",
        "c_err": "Error ŷ − y",
        "c_loss": "Loss",
        "c_chain": "The chain rule, factor by factor",
        "c_grad_out": "Gradients of the output layer",
        "c_grad_hidden": "Gradients of the hidden layer",
        "c_sigma": "Sigmoid and its derivative",
        "c_result": "Result",
        "c_loss_before": "Loss before the step",
        "c_loss_after": "Loss after the step",
        "c_pred_before": "Prediction before the step",
        "c_pred_after": "Prediction after the step",
        "c_notyet": "This step has not run yet — press \"Next step\".",
        "c_stepdone": "Steps completed: {n} of {total}",
        "c_rate": "Learning rate",
        "dir_down": "went down",
        "dir_up": "went up",
        "dir_same": "stayed practically the same",

        "what_hdr": "What happened?",
        "what_down": "The loss fell from {l0} to {l1} (by {d}): the step went in the "
                     "direction of decreasing loss, and there the loss did drop.",
        "what_up": "The loss rose from {l0} to {l1} (by {d}): the direction is right, "
                   "but the step is too large — it jumped past the neighbourhood of the "
                   "minimum. Reduce η.",
        "what_same": "The loss barely changed ({l0} → {l1}): the gradients are small, "
                     "so the parameters moved little.",
        "what_neg": "Parameter {k} has a negative gradient ({g}), so subtracting "
                    "−η·∂L/∂w increased it by {d}.",
        "what_pos": "Parameter {k} has a positive gradient ({g}), so it decreased by {d}.",
        "what_zero": "Parameter {k} has a gradient of {g}: it does not move on this step.",
        "what_worst": "The parameter that changed most is {k}: its gradient has the "
                      "largest magnitude ({g}).",
        "what_sat": "The sigmoid is saturated (ŷ = {yhat}), its derivative σ′(z_o) = "
                    "{dsig_o} is close to zero — that is why all the gradients are so small.",

        "legend_fwd": "Forward pass: left → right",
        "legend_bwd": "Backward pass: right → left",
        "chain_hdr": "Dependency chain for parameter {k}",
        "chain_none": "No chain selected yet",
        "chain_hint": "Click a row in the \"Gradients\" table to see the chain of factors "
                      "for that parameter.",

        "status_ready": "Ready. Press \"Next step\" — it runs exactly one stage.",
        "status_lang": "Interface language switched.",
        "status_step": "Step {n}: {title}",
        "status_first": "This is the first step: there is nowhere to go back.",
        "status_last": "End of the process reached: step {n} is the last one in this mode.",
        "status_reset": "Reset: progress cleared, your parameters kept.",
        "status_example": "Example \"{name}\" loaded: every field updated.",
        "status_full": "Full process shown: {n} stages.",
        "status_bad": "Not a number: \"{raw}\". The last valid value is kept.",
        "status_range": "{name}: allowed from {lo} to {hi}. Nearest allowed value taken.",
        "status_mode": "Mode: {mode}. Steps {a}–{b} are available.",
        "status_selected": "Chain shown for parameter {k}.",

        "guide": GUIDE_EN,
    },
}

# ---------------------------------------------------------------
# 7. Режимы и последовательность шагов
# ---------------------------------------------------------------
# Одиннадцать стадий: десять смысловых из последовательности плюс пересчёт,
# без которого нечего показывать в разделе «до и после».
MODES = {
    "fwd": (1, 7),
    "bwd": (8, 11),
    "full": (1, 11),
}
MODE_LABEL_KEY = {"fwd": "mode_fwd", "bwd": "mode_bwd", "full": "mode_full"}
TOTAL_STEPS = 11

# Узлы и связи, подсвечиваемые на текущей стадии.
STEPS = (
    {"n": 1, "phase": "fwd", "title": "st1_t",
     "nodes": ("X1", "X2"), "edges": ()},
    {"n": 2, "phase": "fwd", "title": "st2_t",
     "nodes": ("H1",), "edges": (("X1", "H1"), ("X2", "H1"))},
    {"n": 3, "phase": "fwd", "title": "st3_t",
     "nodes": ("H1",), "edges": (("H1", "O"),)},
    {"n": 4, "phase": "fwd", "title": "st4_t",
     "nodes": ("H2",), "edges": (("X1", "H2"), ("X2", "H2"))},
    {"n": 5, "phase": "fwd", "title": "st5_t",
     "nodes": ("H2",), "edges": (("H2", "O"),)},
    {"n": 6, "phase": "fwd", "title": "st6_t",
     "nodes": ("O",), "edges": (("H1", "O"), ("H2", "O"))},
    {"n": 7, "phase": "fwd", "title": "st7_t",
     "nodes": ("O", "L"), "edges": (("O", "L"),)},
    {"n": 8, "phase": "bwd", "title": "st8_t",
     "nodes": ("L", "O"), "edges": (("O", "L"), ("H1", "O"), ("H2", "O"))},
    {"n": 9, "phase": "bwd", "title": "st9_t",
     "nodes": ("O", "H1", "H2"),
     "edges": (("H1", "O"), ("H2", "O"), ("X1", "H1"), ("X2", "H1"),
               ("X1", "H2"), ("X2", "H2"))},
    {"n": 10, "phase": "upd", "title": "st10_t",
     "nodes": ("H1", "H2", "O"),
     "edges": (("X1", "H1"), ("X2", "H1"), ("X1", "H2"), ("X2", "H2"),
               ("H1", "O"), ("H2", "O"))},
    {"n": 11, "phase": "upd", "title": "st11_t",
     "nodes": ("O", "L"), "edges": (("H1", "O"), ("H2", "O"), ("O", "L"))},
)
STEP_BY_N = {st["n"]: st for st in STEPS}
# Ключ параметра, чья цепочка подсвечивается по умолчанию на данной стадии.
STEP_FOCUS_PARAM = {8: "w5", 9: "w1", 10: "w5", 11: "w5"}

# ---------------------------------------------------------------
# 8. Расчёт: один шаг обучения целиком и текст каждой стадии
# ---------------------------------------------------------------
class Learn:
    """Один полный учебный шаг: прямой ход, потери, обратный ход, обновление.

    Всё считается один раз из состояния сети и хранится здесь, поэтому панель
    расчёта, граф, таблица градиентов и разбор «до/после» показывают числа из
    одного и того же места и не могут разойтись.
    """

    def __init__(self, state):
        self.s = dict(state)
        self.f = forward(state)
        self.loss = loss_of(self.f, state["y"])
        self.b = backward(state, self.f)
        self.new = updated_state(state, self.b)
        self.f2 = forward(self.new)
        self.loss2 = loss_of(self.f2, state["y"])

    def grad(self, key):
        return grad_of(self.b, key)

    def delta(self, key):
        return delta_of(self.s, self.b, key)

    def new_value(self, key):
        return self.new[key]

    @property
    def loss_change(self):
        return self.loss2 - self.loss


def _hdr(text):
    """Заголовок блока расчёта."""
    return "· " + text


def _param_lines(lrn, keys):
    """Строка «имя = значение (что это)» для каждого параметра."""
    out = []
    for k in keys:
        out.append(f"  {DISP[k]} = {f4(lrn.s[k])}   ({EDGE_HINT[k]})")
    return out


def _chain_lines(lrn, key, upto=None):
    """Множители цепного правила для одного параметра, построчно."""
    chain = chain_of(lrn.s, lrn.f, lrn.b, key)
    if upto is not None:
        chain = chain[:upto]
    out = []
    for item in chain:
        out.append(f"  {item['sym']} = {item['expr']}")
        out.append(f"        = {item['sub']} = {f4(item['val'])}")
        out.append(f"  " + T("c_result") + f": {item['cum_label']} = {f4(item['cum'])}")
    return out


def _b1(lrn):
    s = lrn.s
    out = [_hdr(T("c_inputs")),
           f"  x₁ = {f4(s['x1'])}",
           f"  x₂ = {f4(s['x2'])}",
           f"  y = {f4(s['y'])}   ({T('c_target')})",
           "",
           _hdr(T("c_rate")),
           f"  η = {f4(s['eta'])}",
           "",
           _hdr(T("c_params"))]
    out += _param_lines(lrn, HIDDEN_KEYS)
    out += _param_lines(lrn, OUTPUT_KEYS)
    out += ["", _hdr(T("c_sigma")),
            f"  {FORMULA_SIGMOID}",
            f"  {FORMULA_DSIGMOID}"]
    return out


def _b2(lrn):
    s, f = lrn.s, lrn.f
    return [_hdr(T("c_hidden1")),
            f"  {FORMULA_Z_H1}",
            f"  z_h₁ = {signed(s['x1'])}·{signed(s['w1'])} + {signed(s['x2'])}·{signed(s['w2'])} + {signed(s['b_h1'])}",
            f"  z_h₁ = {f4(f['z_h1'])}"]


def _b3(lrn):
    f = lrn.f
    return [_hdr(T("c_hidden1")),
            f"  h₁ = σ(z_h₁)",
            f"  h₁ = σ({f4(f['z_h1'])})",
            f"  h₁ ≈ {f4(f['h1'])}"]


def _b4(lrn):
    s, f = lrn.s, lrn.f
    return [_hdr(T("c_hidden2")),
            f"  {FORMULA_Z_H2}",
            f"  z_h₂ = {signed(s['x1'])}·{signed(s['w3'])} + {signed(s['x2'])}·{signed(s['w4'])} + {signed(s['b_h2'])}",
            f"  z_h₂ = {f4(f['z_h2'])}"]


def _b5(lrn):
    f = lrn.f
    return [_hdr(T("c_hidden2")),
            f"  h₂ = σ(z_h₂)",
            f"  h₂ = σ({f4(f['z_h2'])})",
            f"  h₂ ≈ {f4(f['h2'])}"]


def _b6(lrn):
    s, f = lrn.s, lrn.f
    return [_hdr(T("c_output")),
            f"  {FORMULA_Z_O}",
            f"  z_o = {signed(f['h1'])}·{signed(s['w5'])} + {signed(f['h2'])}·{signed(s['w6'])} + {signed(s['b_o'])}",
            f"  z_o = {f4(f['z_o'])}"]


def _b7(lrn):
    s, f = lrn.s, lrn.f
    err = f["yhat"] - s["y"]
    return [_hdr(T("c_output")),
            f"  ŷ = σ(z_o)",
            f"  ŷ = σ({f4(f['z_o'])})",
            f"  ŷ ≈ {f4(f['yhat'])}",
            "",
            _hdr(T("c_pred")), f"  ŷ = {f4(f['yhat'])}",
            _hdr(T("c_target")), f"  y = {f4(s['y'])}",
            _hdr(T("c_err")),
            f"  ŷ − y = {f4(f['yhat'])} {MINUS} {f4(s['y'])} = {f4(err)}",
            _hdr(T("c_loss")),
            f"  {FORMULA_LOSS}",
            f"  L = 1/2·({f4(err)})²",
            f"  L = {f4(lrn.loss)}"]


def _b8(lrn):
    out = [_hdr(T("c_grad_out")), _hdr(T("c_chain"))]
    out += _chain_lines(lrn, "w5")
    out += ["", _hdr(T("c_grad_out"))]
    for k in OUTPUT_KEYS:
        sym = GRAD_SYM[k][0]
        last = chain_of(lrn.s, lrn.f, lrn.b, k)[-1]
        out.append(f"  {sym} = ∂L/∂z_o · {last['sym']}")
        out.append(f"        = {term(lrn.b['dL_dzo'])} · {last['sub']} = {f4(lrn.grad(k))}")
    return out


def _b9(lrn):
    out = [_hdr(T("c_grad_hidden"))]
    for hnum, keys in (("1", ("w1", "w2", "b_h1")), ("2", ("w3", "w4", "b_h2"))):
        out.append("")
        out.append(_hdr(T("c_hidden" + hnum)))
        out.append(f"  ∂L/∂h{hnum} = ∂L/∂z_o · ∂z_o/∂h{hnum}"
                   f" = {term(lrn.b['dL_dzo'])} · {term(lrn.s['w5' if hnum == '1' else 'w6'])}"
                   f" = {f4(lrn.b['dL_dh' + hnum])}")
        out.append(f"  ∂h{hnum}/∂z_h{hnum} = h{hnum}·(1 − h{hnum})"
                   f" = {f4(lrn.f['h' + hnum])}·(1 {MINUS} {f4(lrn.f['h' + hnum])})"
                   f" = {f4(lrn.b['dsig_h' + hnum])}")
        out.append(f"  ∂L/∂z_h{hnum} = ∂L/∂h{hnum} · ∂h{hnum}/∂z_h{hnum}"
                   f" = {f4(lrn.b['dL_dh' + hnum])} · {f4(lrn.b['dsig_h' + hnum])}"
                   f" = {f4(lrn.b['dL_dzh' + hnum])}")
        for k in keys:
            last = chain_of(lrn.s, lrn.f, lrn.b, k)[-1]
            out.append(f"  {GRAD_SYM[k][0]} = ∂L/∂z_h{hnum} · {last['sym']}"
                       f" = {term(lrn.b['dL_dzh' + hnum])} · {last['sub']}"
                       f" = {f4(lrn.grad(k))}")
    return out


def _b10(lrn):
    out = [_hdr(T("c_update")),
          f"  {FORMULA_UPDATE}",
          f"  η = {f4(lrn.s['eta'])}"]
    for k in WEIGHT_KEYS:
        g = lrn.grad(k)
        d = lrn.delta(k)
        out += ["",
                f"  {DISP[k]} = {f4(lrn.s[k])}   ({EDGE_HINT[k]})",
                f"  ∂L/∂{DISP[k]} = {f4(g)}",
                f"  Δ{DISP[k]} = −η · ∂L/∂{DISP[k]} = −{f4(lrn.s['eta'])} · {term(g)} = {f4(d)}",
                f"  {DISP[k]}_new = {f4(lrn.s[k])} {MINUS} {f4(lrn.s['eta'])}·({f4(g)}) = {f4(lrn.new_value(k))}"]
    return out


def _b11(lrn):
    s, f2 = lrn.s, lrn.f2
    out = [_hdr(T("c_recalc")),
           _hdr(T("c_pred_before")), f"  ŷ = {f4(lrn.f['yhat'])}",
           _hdr(T("c_loss_before")), f"  L = {f4(lrn.loss)}",
           "",
           _hdr(T("c_pred_after")),
           f"  z_h₁ = {f4(s['x1'])}·{f4(lrn.new['w1'])} + {f4(s['x2'])}·{f4(lrn.new['w2'])} + {f4(lrn.new['b_h1'])} = {f4(f2['z_h1'])}",
           f"  h₁ = σ({f4(f2['z_h1'])}) ≈ {f4(f2['h1'])}",
           f"  z_h₂ = {f4(s['x1'])}·{f4(lrn.new['w3'])} + {f4(s['x2'])}·{f4(lrn.new['w4'])} + {f4(lrn.new['b_h2'])} = {f4(f2['z_h2'])}",
           f"  h₂ = σ({f4(f2['z_h2'])}) ≈ {f4(f2['h2'])}",
           f"  z_o = {f4(f2['h1'])}·{f4(lrn.new['w5'])} + {f4(f2['h2'])}·{f4(lrn.new['w6'])} + {f4(lrn.new['b_o'])} = {f4(f2['z_o'])}",
           f"  ŷ = σ({f4(f2['z_o'])}) ≈ {f4(f2['yhat'])}",
           f"  L = 1/2·({f4(f2['yhat'])} {MINUS} {f4(s['y'])})² = {f4(lrn.loss2)}",
           "",
           _hdr(T("c_result")),
           f"  ΔL = {f4(lrn.loss)} → {f4(lrn.loss2)}   ({signed(lrn.loss_change)})"]
    return out


STEP_BLOCKS = {1: _b1, 2: _b2, 3: _b3, 4: _b4, 5: _b5, 6: _b6, 7: _b7,
               8: _b8, 9: _b9, 10: _b10, 11: _b11}


def step_head(n):
    """Заголовок стадии в панели расчёта."""
    return f"{'=' * 4} {T('lbl_step', n=n, total=TOTAL_STEPS)} · {T(STEP_BY_N[n]['title'])} {'=' * 4}"


def calc_text(lrn, upto):
    """Весь пройденный расчёт: стадии 1..upto подряд, с сохранением порядка."""
    if lrn is None or upto < 1:
        return T("c_notyet")
    lines = []
    for n in range(1, upto + 1):
        lines.append(step_head(n))
        lines.extend(STEP_BLOCKS[n](lrn))
        lines.append("")
    lines.append(T("c_stepdone", n=upto, total=TOTAL_STEPS))
    return "\n".join(lines)


def what_happened(lrn):
    """Разбор «Что произошло?» — только по фактическим числам этого шага."""
    if lrn is None:
        return ""
    lines = []
    d = lrn.loss_change
    rel = abs(d) / max(lrn.loss, 1e-12)
    if abs(d) < 1e-9 or rel < 1e-4:
        lines.append(T("what_same", l0=f4(lrn.loss), l1=f4(lrn.loss2)))
    elif d < 0:
        lines.append(T("what_down", l0=f4(lrn.loss), l1=f4(lrn.loss2), d=f4(abs(d))))
    else:
        lines.append(T("what_up", l0=f4(lrn.loss), l1=f4(lrn.loss2), d=f4(abs(d))))
    # Два параметра с наибольшим сдвигом — чтобы разбор не превращался в простыню
    order = sorted(WEIGHT_KEYS, key=lambda k: -abs(lrn.delta(k)))
    for k in order[:2]:
        g, dl = lrn.grad(k), lrn.delta(k)
        if abs(g) < 5e-9:
            lines.append(T("what_zero", k=DISP[k], g=f4(g)))
        elif g < 0:
            lines.append(T("what_neg", k=DISP[k], g=f4(g), d=f4(abs(dl))))
        else:
            lines.append(T("what_pos", k=DISP[k], g=f4(g), d=f4(abs(dl))))
    lines.append(T("what_worst", k=DISP[order[0]], g=f4(lrn.grad(order[0]))))
    if lrn.b["dsig_o"] < 0.02:
        lines.append(T("what_sat", yhat=f4(lrn.f["yhat"]), dsig_o=f4(lrn.b["dsig_o"])))
    return "\n".join("• " + ln for ln in lines)


def why_now(lrn, n):
    """Короткое пояснение к текущей стадии, с числами именно этого шага."""
    if lrn is None or n < 1:
        return T("c_notyet")
    s, f, b = lrn.s, lrn.f, lrn.b
    err = f["yhat"] - s["y"]
    args = {
        "x1": f4(s["x1"]), "x2": f4(s["x2"]), "z1": f4(f["z_h1"]), "z2": f4(f["z_h2"]),
        "h1": f4(f["h1"]), "h2": f4(f["h2"]), "zo": f4(f["z_o"]),
        "yhat": f4(f["yhat"]), "y": f4(s["y"]), "err": f4(err), "loss": f4(lrn.loss),
        "dL_dyhat": f4(b["dL_dyhat"]), "dsig_o": f4(b["dsig_o"]), "dL_dzo": f4(b["dL_dzo"]),
        "dL_dh1": f4(b["dL_dh1"]), "dL_dh2": f4(b["dL_dh2"]),
        "dL_dzh1": f4(b["dL_dzh1"]), "dL_dzh2": f4(b["dL_dzh2"]),
        "eta": f4(s["eta"]), "yhat0": f4(f["yhat"]), "yhat1": f4(lrn.f2["yhat"]),
        "l0": f4(lrn.loss), "l1": f4(lrn.loss2),
        "dir_word": T("dir_down") if lrn.loss_change < -1e-12 else (
            T("dir_up") if lrn.loss_change > 1e-12 else T("dir_same")),
    }
    return T("why" + str(n), **args)

# ---------------------------------------------------------------
# 9. Поле ввода одного параметра
# ---------------------------------------------------------------
class ParamField(ttk.Frame):
    """Подпись + поле ввода одного числа.

    Нечисловой ввод не роняет программу и не попадает в расчёт: поле
    подсвечивается, показывается понятное сообщение, а модель продолжает
    использовать последнее правильное значение. Запятая как разделитель
    допускается наравне с точкой.
    """

    def __init__(self, parent, key, value, on_change):
        super().__init__(parent)
        self.key = key
        self.lo, self.hi = LIMITS[key]
        self.on_change = on_change
        self._lock = False
        self._bad = False

        self.lbl = ttk.Label(self, text=DISP[key], width=4, anchor="w")
        self.lbl.pack(side="left")
        self.hint = ttk.Label(self, text=EDGE_HINT[key], width=9, anchor="w",
                              foreground="#6a6a63")
        self.hint.pack(side="left")
        self.var = tk.StringVar(value=fmt(value))
        self.entry = ttk.Entry(self, textvariable=self.var, width=9, justify="right")
        self.entry.pack(side="right")
        self.var.trace_add("write", self._changed)

    def set_value(self, v):
        """Программная установка числа (пример, сброс) без вызова on_change."""
        self._lock = True
        self.var.set(fmt(v))
        self._mark(False)
        self._lock = False

    def _mark(self, bad):
        self._bad = bad
        self.entry.configure(style="BadEntry.TEntry" if bad else "TEntry")

    def _changed(self, *_):
        if self._lock:
            return
        raw = self.var.get().strip()
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            self._mark(True)
            self.on_change(None, raw, self.key)
            return
        if not math.isfinite(v):
            self._mark(True)
            self.on_change(None, raw, self.key)
            return
        clamped = min(self.hi, max(self.lo, v))
        out = abs(clamped - v) > 1e-12
        self._mark(out)
        self.on_change(clamped, raw, self.key, out)

    def focus_self(self):
        self.entry.focus_set()
        self.entry.select_range(0, "end")


# ---------------------------------------------------------------
# 10. Граф сети: он строится из модели, а не из картинки
# ---------------------------------------------------------------
class NetworkView(ttk.Frame):
    """Вычислительный граф 2 → 2 → 1 и полоса цепочки зависимостей под ним.

    Прямой ход читается слева направо, обратный — справа налево: направления
    подписаны и показаны разными стрелками, а текущая стадия подсвечена.
    """

    R = 27                     # радиус узла
    CHAIN_BAND = 118           # высота полосы цепочки

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, bg=COL_PAPER, highlightthickness=1,
                                highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    # ---------- вспомогательное ----------
    def _arrow(self, x1, y1, x2, y2, color, width=2, dash=None, head=True):
        c = self.canvas
        c.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash,
                      capstyle="round")
        if not head:
            return
        ang = math.atan2(y2 - y1, x2 - x1)
        for da, scale in ((math.radians(152), 1.0), (math.radians(-152), 1.0)):
            c.create_line(x2, y2, x2 + 11 * scale * math.cos(ang + da),
                          y2 + 11 * scale * math.sin(ang + da),
                          fill=color, width=width, capstyle="round")

    def _text(self, x, y, s, **kw):
        kw.setdefault("fill", COL_TEXT)
        return self.canvas.create_text(x, y, text=s, **kw)

    def _node(self, key, x, y, name, value, active, font, sub="", sub_side="c"):
        """Кружок нейрона: имя, значение, при необходимости — свёртка внутри.

        sub_side задает, куда убирать подпись: "c" — под узел (входам), "w" —
        вправо (нейронам скрытого слоя), "e" — влево (выходу). Под узлом подписи
        входов и весов пересекаются, поэтому каждый слой уходит в свою сторону.
        """
        c = self.canvas
        fill = COL_ACTIVE_FILL if active else COL_NODE
        outline = COL_ACTIVE if active else COL_NODE_LINE
        c.create_oval(x - self.R, y - self.R, x + self.R, y + self.R,
                      fill=fill, outline=outline,
                      width=3 if active else 1)
        self._text(x, y - 7, name, font=font["mid"])
        if value:
            self._text(x, y + 9, value, font=font["small"])
        if not sub:
            return
        if sub_side == "w":
            self._text(x + self.R + 8, y + self.R, sub, anchor="w",
                       font=font["small"], fill="#5a5a53")
        elif sub_side == "e":
            self._text(x - self.R - 8, y + self.R, sub, anchor="e",
                       font=font["small"], fill="#5a5a53")
        else:
            self._text(x, y + self.R + 12, sub, font=font["small"],
                       fill="#5a5a53")

    # ---------- основной рисунок ----------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 560)
        h = max(c.winfo_height(), 400)
        app = self.app
        lrn = app.lrn
        s, f, b = lrn.s, lrn.f, lrn.b
        step = app.step
        st = STEP_BY_N.get(step)
        nodes = set(st["nodes"]) if st else set()
        edges = set(st["edges"]) if st else set()
        back = step >= 8
        upd = step >= 10
        # В минимально допустимом окне места впритык: подписи свёрток уходят,
        # потому что те же числа всегда есть в панели расчёта, а налезать на
        # подписи весов им некуда.
        compact = w < 640

        top, bottom = 52, h - self.CHAIN_BAND - 16
        gh = bottom - top
        cols = {"X": 78, "H": 78 + (w - 170) * 0.34,
                "O": 78 + (w - 170) * 0.68, "L": w - 92}
        pos = {
            "X1": (cols["X"], top + gh * 0.28),
            "X2": (cols["X"], top + gh * 0.76),
            "H1": (cols["H"], top + gh * 0.22),
            "H2": (cols["H"], top + gh * 0.82),
            "O": (cols["O"], top + gh * 0.52),
            "L": (cols["L"], top + gh * 0.52),
        }
        self.pos = pos
        font = app.fonts

        # --- подписи направлений ---
        self._text(14, 20, T("legend_fwd"), anchor="w", font=font["small"],
                   fill=COL_FWD)
        self._text(w - 14, 20, T("legend_bwd"), anchor="e", font=font["small"],
                   fill=COL_BWD)
        if compact and back and app.selected and self._grad_known(step, app.selected):
            # В узком окне градиент выбранного параметра пишется отдельной
            # строкой: на ребре для неё места уже нет.
            self._text(14, 42, f"∂L/∂{DISP[app.selected]} = {f4(lrn.grad(app.selected))}",
                       anchor="w", font=font["smallb"], fill=COL_BWD)

        # --- связи ---
        links = (("X1", "H1", "w1"), ("X2", "H1", "w2"),
                 ("X1", "H2", "w3"), ("X2", "H2", "w4"),
                 ("H1", "O", "w5"), ("H2", "O", "w6"))
        for a, nd, key in links:
            x1, y1 = pos[a]
            x2, y2 = pos[nd]
            hot = (a, nd) in edges
            color = COL_ACTIVE if hot else (COL_FWD if not back else "#7fa0ba")
            self._arrow(x1 + self.R, y1, x2 - self.R, y2, color,
                        3 if hot else 2)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # вес: после обновления показываем и старое, и новое значение;
            # в узком окне — только новое, иначе подпись лезет на кружок.
            if upd:
                lab = (f"{DISP[key]}={f2(lrn.new_value(key))}" if compact else
                       f"{DISP[key]} {f2(s[key])} → {f2(lrn.new_value(key))}")
                lcol = COL_POS
            else:
                lab = f"{DISP[key]}={f2(s[key])}" if compact \
                    else f"{DISP[key]} = {f2(s[key])}"
                lcol = COL_TEXT
            self._text(mx, my - 12, lab, font=font["smallb"], fill=lcol)
            # Градиент подписываем только у выбранного параметра: девять таких
            # подписей сразу превращали граф в кашу. В узком окне подпись
            # с ребра уходит вовсе — её заменяет строка выбранного параметра
            # под легендой.
            if (back and not compact and key == app.selected
                    and self._grad_known(step, key)):
                self._text(mx, my + 15, f"∂L/∂{DISP[key]} = {f4(lrn.grad(key))}",
                           font=font["tiny"], fill=COL_BWD)
            # обратная стрелка — параллельно, чуть ниже
            if back:
                self._arrow(x2 - self.R, y2 + 9, x1 + self.R, y1 + 9,
                            COL_BWD if hot else "#d8a7a2",
                            2 if hot else 1, dash=(5, 4))

        # --- связь ŷ → L ---
        x1, y1 = pos["O"]
        x2, y2 = pos["L"]
        hot = ("O", "L") in edges
        self._arrow(x1 + self.R, y1, x2 - 46, y2,
                    COL_ACTIVE if hot else "#7fa0ba", 3 if hot else 2)
        # Отдельная буква "L" над этой стрелкой не нужна: сам блок потерь
        # подписан, а подпись только падала на b_o.

        # --- узлы ---
        self._node("X1", *pos["X1"], "x₁", f2(s["x1"]), "X1" in nodes, font,
                   sub="" if compact else EDGE_HINT["w1"])
        self._node("X2", *pos["X2"], "x₂", f2(s["x2"]), "X2" in nodes, font,
                   sub="" if compact else EDGE_HINT["w3"])
        show_h1 = step >= 3
        show_h2 = step >= 5
        self._node("H1", *pos["H1"], "h₁", f4(f["h1"]) if show_h1 else "—",
                   "H1" in nodes, font, sub_side="w",
                   sub=f"z_h₁ = {f4(f['z_h1'])}" if show_h1 and not compact else "")
        self._node("H2", *pos["H2"], "h₂", f4(f["h2"]) if show_h2 else "—",
                   "H2" in nodes, font, sub_side="w",
                   sub=f"z_h₂ = {f4(f['z_h2'])}" if show_h2 and not compact else "")
        self._node("O", *pos["O"], "ŷ", f4(f["yhat"]) if step >= 6 else "—",
                   "O" in nodes, font, sub_side="e",
                   sub=f"z_o = {f4(f['z_o'])}" if step >= 6 and not compact else "")
        # смещения — отдельными подписями у своих нейронов
        for nd, key in (("H1", "b_h1"), ("H2", "b_h2"), ("O", "b_o")):
            x, y = pos[nd]
            val = f"{DISP[key]} = {f2(s[key])}"
            if upd:
                val += f" → {f2(lrn.new_value(key))}"
            self._text(x + self.R + 6, y - self.R - 4, val, anchor="w",
                       font=font["tiny"], fill=COL_POS if upd else "#5a5a53")
        # узел потерь
        lx, ly = pos["L"]
        box_col = COL_ACTIVE if "L" in nodes else COL_NODE_LINE
        c.create_rectangle(lx - 46, ly - 26, lx + 46, ly + 26,
                           fill=COL_ACTIVE_FILL if "L" in nodes else COL_NODE,
                           outline=box_col, width=3 if "L" in nodes else 1)
        self._text(lx, ly - 8, "L", font=font["mid"])
        self._text(lx, ly + 10, f4(lrn.loss) if step >= 7 else "—",
                   font=font["small"])
        if upd and step >= 11:
            self._text(lx, ly + 40, f"→ {f4(lrn.loss2)}", font=font["tiny"],
                       fill=COL_POS)

        self._draw_chain(w, h, font)

    @staticmethod
    def _grad_known(step, key):
        """Градиенты выходного слоя известны с 8-й стадии, скрытого — с 9-й."""
        if key in OUTPUT_KEYS:
            return step >= 8
        return step >= 9

    # ---------- полоса цепочки зависимостей ----------
    def _draw_chain(self, w, h, font):
        c = self.canvas
        app = self.app
        lrn = app.lrn
        top = h - self.CHAIN_BAND + 8
        c.create_line(12, top - 10, w - 12, top - 10, fill=COL_FRAME)
        key = app.selected
        if key is None:
            self._text(14, top + 8, T("chain_none"), anchor="w",
                       font=font["sub"], fill=COL_DIM)
            self._text(14, top + 30, T("chain_hint"), anchor="w",
                       font=font["tiny"], fill=COL_DIM)
            return
        chain = chain_of(lrn.s, lrn.f, lrn.b, key)
        self._text(14, top + 8, T("chain_hdr", k=DISP[key]), anchor="w",
                   font=font["midb"], fill=COL_CHAIN)
        # Блоки идут справа налево: цепное правило начинает от потерь и доходит
        # до параметра, и направление здесь — часть учебного смысла.
        n = len(chain)
        avail = w - 40
        bw = min(150.0, (avail - 14 * (n - 1)) / n)
        gap = (avail - bw * n) / max(n - 1, 1)
        x_right = w - 20
        cy = top + 48
        need = 9 if key in HIDDEN_KEYS else 8
        active = app.step >= need
        for i, item in enumerate(chain):
            x = x_right - i * (bw + gap)
            fill = COL_ACTIVE_FILL if active else "#f2f2ee"
            out = COL_ACTIVE if (active and i == len(chain) - 1) else COL_CHAIN
            c.create_rectangle(x - bw, cy - 20, x, cy + 20, fill=fill,
                               outline=out, width=2 if active else 1)
            self._text(x - bw / 2.0, cy - 8, item["sym"], font=font["small"],
                       fill=COL_TEXT)
            # Пока обратный ход не дошёл до этого множителя, число не показываем:
            # иначе полоса выдавала бы ответ раньше, чем студент его получил.
            self._text(x - bw / 2.0, cy + 8,
                       f4(item["val"]) if active else "—",
                       font=font["smallb"],
                       fill=COL_TEXT if active else COL_DIM)
            # Бегущее произведение подписываем под каждым блоком, когда места
            # хватает; в узком окне остаётся только итог — сам градиент.
            cum_txt = ""
            if active and (w >= 640 or i == n - 1):
                cum_txt = f"{item['cum_label']} = {f4(item['cum'])}"
            self._text(x - bw / 2.0, cy + 32, cum_txt,
                       font=font["tiny"], fill=COL_BWD)
            if i < n - 1:
                self._arrow(x - bw, cy, x - bw - gap + 2, cy,
                            COL_CHAIN if active else COL_DIM, 2)
                self._text(x - bw - gap / 2.0, cy - 12, "·", font=font["mid"],
                           fill=COL_CHAIN)

# ---------------------------------------------------------------
# 10b. Прокручиваемая колонка: содержимое не обязано влезать в окно
# ---------------------------------------------------------------
class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемый контейнер для левого столбца.

    Без него при невысоком окне нижние группы параметров просто не
    отрисовываются: Tk не умеет «сжимать» pack, он обрезает столбцы. Колесо мыши
    листает колонку только когда курсор над ней, чтобы не мешать таблице и
    панелям расчёта.
    """

    def __init__(self, master, width=320):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=COL_BG,
                                width=width)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", self._rescroll)
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, self._on_wheel)

    def _rescroll(self, *_):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_wheel(self, event):
        c = self.canvas
        x, y = c.winfo_pointerxy()
        rx, ry = c.winfo_rootx(), c.winfo_rooty()
        if not (rx <= x < rx + c.winfo_width() and ry <= y < ry + c.winfo_height()):
            return
        up = getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4
        c.yview_scroll(-1 if up else 1, "units")


# ---------------------------------------------------------------
# 11. Окно
# ---------------------------------------------------------------
class BackpropApp(tk.Tk):
    """Слева примеры, режим и все параметры; в центре граф сети с цепочкой
    зависимостей; справа вкладки: расчёт, таблица градиентов, пояснения и
    справка. Внизу — строка состояния и имя автора."""

    def __init__(self):
        super().__init__()
        self.title(T("app_title"))
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        # И начальный размер, и нижняя граница зависят от экрана: на ноутбуке
        # 1366x768 окно не должно оказываться больше рабочего стола.
        self._geo_w = min(1520, sw - 40)
        self._geo_h = min(940, sh - 60)
        self.geometry(f"{self._geo_w}x{self._geo_h}")
        self.minsize(min(1300, sw - 40), min(780, sh - 60))

        # ---- состояние ----
        self.state = dict(DEFAULT_EXAMPLE["state"])
        self.step = 0
        self.mode = "full"
        self.selected = "w5"
        self.lrn = Learn(self.state)
        self._text_refs = []
        self._tabs = []
        self._radios = []
        self._silent = False

        self.var_lang = tk.StringVar(value=LANG)
        self.var_mode = tk.StringVar(value=self.mode)
        self.var_status = tk.StringVar()

        self._make_fonts()
        self._make_styles()
        self._build_header()
        self._build_left()
        self._build_center()
        self._build_right()
        self._build_bottom()
        self.refresh()
        self.set_status("status_ready")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ---------- оформление ----------
    def _make_fonts(self):
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")
        self.fonts = {
            "title": tkfont.Font(family=base, size=17, weight="bold"),
            "sub": tkfont.Font(family=base, size=14),
            "mid": tkfont.Font(family=base, size=15, weight="bold"),
            "midb": tkfont.Font(family=base, size=13, weight="bold"),
            "small": tkfont.Font(family=base, size=12),
            "smallb": tkfont.Font(family=base, size=12, weight="bold"),
            "tiny": tkfont.Font(family=base, size=10),
            "mono": tkfont.Font(family=mono, size=12),
            "monob": tkfont.Font(family=mono, size=12, weight="bold"),
            "big": tkfont.Font(family=base, size=20, weight="bold"),
        }
        self.option_add("*Font", self.fonts["sub"])
        self.configure(bg=COL_BG)

    def _make_styles(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=COL_BG, foreground=COL_TEXT)
        st.configure("TFrame", background=COL_BG)
        st.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        st.configure("TLabelframe", background=COL_BG, bordercolor=COL_FRAME)
        st.configure("TLabelframe.Label", background=COL_BG, foreground="#3a3a35",
                     font=self.fonts["midb"])
        st.configure("TNotebook", background=COL_BG, bordercolor=COL_FRAME)
        st.configure("TNotebook.Tab", padding=(12, 6), font=self.fonts["sub"])
        st.configure("TButton", padding=(10, 6), font=self.fonts["sub"])
        st.configure("Go.TButton", font=self.fonts["midb"])
        st.configure("TRadiobutton", background=COL_BG, font=self.fonts["sub"])
        st.map("TNotebook.Tab", background=[("selected", COL_PAPER)],
               foreground=[("selected", COL_TEXT)])
        # Подсветка поля с некорректным вводом: style-имя обязано быть
        # «Имя.TEntry», иначе ttk его не подхватит.
        st.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                     foreground=COL_BAD_FG)

    # ---------- локализация ----------
    def _reg(self, widget, key, attr="text", **kw):
        """Надпись берётся из TEXTS и будет переведена при смене языка."""
        widget.configure(**{attr: T(key, **kw)})
        self._text_refs.append((widget, key, attr, kw))
        return widget

    def switch_language(self, code):
        if code == LANG:
            return
        set_language(code)
        self.retranslate()
        self.set_status("status_lang")

    def retranslate(self):
        """Перевести статические надписи и заново нарисовать динамические."""
        self.title(T("app_title"))
        for widget, key, attr, kw in self._text_refs:
            widget.configure(**{attr: T(key, **kw)})
        for tab, key in self._tabs:
            self.nb.tab(tab, text=T(key))
        for radio, key, _ in self._radios:
            radio.configure(text=T(key))
        for key, field in self.fields.items():
            field.lbl.configure(text=DISP[key])
            field.hint.configure(text=EDGE_HINT[key])
        # Список примеров и его подпись — динамические: без этого в EN
        # оставались русские названия примеров.
        cur = self.cmb_ex.current()
        self.cmb_ex.configure(values=[T(i + "_name") for i in self.ex_ids])
        self.cmb_ex.current(cur)
        self.lbl_ex_desc.configure(text=T(self.ex_ids[cur] + "_desc"))
        for key, lbl in self.step_dots.items():
            lbl.configure(text=str(key))
        self.btn_next.configure(text=T("btn_next"))
        self.btn_prev.configure(text=T("btn_prev"))
        self.btn_reset.configure(text=T("btn_reset"))
        self.btn_full.configure(text=T("btn_full"))
        # Динамические тексты, которые не висят в _text_refs: их тоже надо
        # переводить, иначе после RU -> EN в них остаётся русский.
        self.var_mode_h.set(T("mode_" + self.mode + "_hint"))
        self._fill_guide()
        self._set_headings()
        self.refresh()

    def set_status(self, key, **kw):
        self.var_status.set(T(key, **kw))

    # ---------- построение ----------
    def _build_header(self):
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 2))
        self._reg(ttk.Label(head, font=self.fonts["title"]), "app_title").pack(side="left")
        self._reg(ttk.Label(head, font=self.fonts["sub"], foreground="#4a4a4a"),
                  "subtitle").pack(side="left", padx=(16, 0))
        box = ttk.Frame(head)
        box.pack(side="right")
        self._reg(ttk.Label(box, font=self.fonts["sub"]), "lang_label").pack(side="left")
        for code in ("ru", "en"):
            r = ttk.Radiobutton(box, text=T("lang_" + code), value=code,
                                variable=self.var_lang,
                                command=lambda c=code: self.switch_language(c))
            r.pack(side="left", padx=(8, 0))
            self._radios.append((r, "lang_" + code, code))
        self.columnconfigure(0, minsize=320, weight=0)
        # weight 3:2 — лишняя ширина достаётся графу, но панель расчётов тоже
        # растёт; иначе при раздвижении окна справа образовывалась пустота.
        # Ширина графа не должна уходить под тот размер, при котором подписи
        # начинают ложиться друг на друга: колонка ниже 560 px не сжимается.
        self.columnconfigure(1, minsize=560, weight=3)
        self.columnconfigure(2, minsize=400, weight=2)
        self.rowconfigure(1, weight=1)

    def _build_left(self):
        # Колонка прокручивается: при невысоком окне иначе обрезается нижняя
        # группа параметров и вся полоса шагов.
        scroll = ScrollFrame(self, width=320)
        scroll.grid(row=1, column=0, sticky="nsw")
        left = scroll.inner

        ex_box = ttk.LabelFrame(left, padding=(8, 6))
        self._reg(ex_box, "box_examples")
        ex_box.pack(fill="x", pady=(0, 8))
        self.ex_ids = list(EXAMPLE_IDS)
        self.cmb_ex = ttk.Combobox(ex_box, state="readonly", width=28,
                                   values=[T(i + "_name") for i in self.ex_ids],
                                   font=self.fonts["sub"])
        self.cmb_ex.current(0)
        self.cmb_ex.pack(fill="x")
        self.cmb_ex.bind("<<ComboboxSelected>>", self._on_example_pick)
        self.lbl_ex_desc = ttk.Label(ex_box, font=self.fonts["tiny"],
                                     foreground="#5a5a53", wraplength=288)
        self.lbl_ex_desc.configure(text=T(self.ex_ids[0] + "_desc"))
        self.lbl_ex_desc.pack(anchor="w", pady=(3, 0))

        mode_box = ttk.LabelFrame(left, padding=(8, 6))
        self._reg(mode_box, "box_mode")
        mode_box.pack(fill="x", pady=(0, 8))
        self.var_mode_h = tk.StringVar(value=T("mode_full_hint"))
        for m in ("fwd", "bwd", "full"):
            r = ttk.Radiobutton(mode_box, text=T("mode_" + m), value=m,
                                variable=self.var_mode,
                                command=lambda mm=m: self.set_mode(mm))
            r.pack(anchor="w")
            self._radios.append((r, "mode_" + m, m))
        ttk.Label(mode_box, textvariable=self.var_mode_h, wraplength=288,
                  justify="left", font=self.fonts["tiny"],
                  foreground="#5a5a53").pack(anchor="w", pady=(4, 0))

        self.fields = {}
        # η живёт в одной группе с данными: две отдельные рамки не влезали по
        # высоте, а нижняя группа параметров обрезалась.
        groups = (("box_data", ("x1", "x2", "y", "eta")),
                  ("box_hidden", HIDDEN_KEYS),
                  ("box_output", OUTPUT_KEYS))
        for gkey, keys in groups:
            gb = ttk.LabelFrame(left, padding=(8, 4))
            self._reg(gb, gkey)
            gb.pack(fill="x", pady=(0, 4))
            for k in keys:
                fld = ParamField(gb, k, self.state[k], self.on_param)
                fld.pack(fill="x", pady=1)
                self.fields[k] = fld
        self._reg(ttk.Label(left, font=self.fonts["tiny"], foreground="#5a5a53",
                            wraplength=296), "input_hint").pack(anchor="w", pady=(2, 0))

    def _on_example_pick(self, *_):
        self.apply_example(self.ex_ids[self.cmb_ex.current()])

    def _build_center(self):
        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        # Полоса шагов — под графом: она важнее, чем ещё одна группа слева, и
        # всегда должна оставаться на виду.
        strip = ttk.LabelFrame(center, padding=(10, 6))
        self._reg(strip, "box_progress")
        strip.pack(side="bottom", fill="x", pady=(6, 0))
        self.var_step = tk.StringVar()
        top = ttk.Frame(strip)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.var_step,
                  font=self.fonts["midb"]).pack(side="left")
        dots = ttk.Frame(top)
        dots.pack(side="left", padx=(14, 0))
        self.step_dots = {}
        for n in range(1, TOTAL_STEPS + 1):
            lbl = tk.Label(dots, text=str(n), width=2, bg="#eceae4",
                           fg="#6a6a63", font=self.fonts["smallb"], bd=1,
                           relief="solid", cursor="hand2")
            lbl.pack(side="left", padx=1)
            lbl.bind("<Button-1>", lambda e, nn=n: self.goto_step(nn))
            self.step_dots[n] = lbl
        # Кнопки — двумя рядами по две: в один ряд они не умещаются даже при
        # минимальной ширине окна, и последняя кнопка просто исчезала.
        r1 = ttk.Frame(strip)
        r1.pack(fill="x", pady=(6, 0))
        self.btn_next = ttk.Button(r1, text=T("btn_next"), style="Go.TButton",
                                   command=self.next_step)
        self.btn_next.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_prev = ttk.Button(r1, text=T("btn_prev"), command=self.prev_step)
        self.btn_prev.pack(side="left", expand=True, fill="x")
        r2 = ttk.Frame(strip)
        r2.pack(fill="x", pady=(4, 0))
        self.btn_reset = ttk.Button(r2, text=T("btn_reset"), command=self.reset)
        self.btn_reset.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_full = ttk.Button(r2, text=T("btn_full"), command=self.show_full)
        self.btn_full.pack(side="left", expand=True, fill="x")
        self.lbl_step_name = ttk.Label(strip, font=self.fonts["sub"],
                                       foreground=COL_FWD, wraplength=420,
                                       justify="left")
        self.lbl_step_name.pack(anchor="w", pady=(4, 0))
        # Длина подписи шага разная в RU и EN, поэтому перелом по словам
        # подстраивается под фактическую ширину полосы.
        strip.bind("<Configure>", lambda e: self.lbl_step_name.configure(
            wraplength=max(e.width - 24, 200)))
        self.view = NetworkView(center, self)
        self.view.pack(side="top", fill="both", expand=True)

    def _text_panel(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        # width=1 обязательно: иначе дефолтные 80 символов заставляют tk.Text
        # требовать ~860 px и обрезать всю правую колонку.
        widget = tk.Text(frame, wrap="word", width=1, height=1,
                         bg=COL_PAPER, fg=COL_TEXT,
                         highlightthickness=1, highlightbackground=COL_FRAME,
                         bd=0, padx=10, pady=8, font=self.fonts["mono"],
                         spacing3=3, cursor="arrow")
        bar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=bar.set, state="disabled")
        bar.pack(side="right", fill="y")
        widget.pack(side="left", fill="both", expand=True)
        return widget

    def _build_right(self):
        right = ttk.Frame(self, padding=(6, 4, 12, 4))
        right.grid(row=1, column=2, sticky="nsew")
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.tab_calc = ttk.Frame(self.nb)
        self.tab_grads = ttk.Frame(self.nb)
        self.tab_why = ttk.Frame(self.nb)
        self.tab_guide = ttk.Frame(self.nb)
        self.nb.add(self.tab_calc, text=T("tab_calc"))
        self.nb.add(self.tab_grads, text=T("tab_grads"))
        self.nb.add(self.tab_why, text=T("tab_why"))
        self.nb.add(self.tab_guide, text=T("tab_guide"))
        for tab, key in ((self.tab_calc, "tab_calc"), (self.tab_grads, "tab_grads"),
                         (self.tab_why, "tab_why"), (self.tab_guide, "tab_guide")):
            self._tabs.append((tab, key))

        self.txt_calc = self._text_panel(self.tab_calc)

        gb = ttk.Frame(self.tab_grads)
        gb.pack(fill="both", expand=True)
        cols = ("param", "value", "grad", "delta", "new")
        self.tree = ttk.Treeview(gb, columns=cols, show="headings", height=9)
        self._set_headings()
        widths = (66, 74, 62, 76, 62)
        for cid, wd in zip(cols, widths):
            self.tree.column(cid, width=wd, anchor="e", stretch=True)
        sbar = ttk.Scrollbar(gb, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sbar.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.txt_before = self._text_panel(self.tab_grads)
        self.txt_before.configure(font=self.fonts["small"])

        self.txt_why = self._text_panel(self.tab_why)
        self.txt_why.configure(font=self.fonts["sub"], wrap="word")
        self.txt_guide = self._text_panel(self.tab_guide)
        self.txt_guide.configure(font=self.fonts["sub"])
        self._fill_guide()

    def _set_headings(self):
        for cid, key in (("param", "th_param"), ("value", "th_value"),
                         ("grad", "th_grad"), ("delta", "th_delta"),
                         ("new", "th_new")):
            self.tree.heading(cid, text=T(key))

    def _fill_guide(self):
        self._set_text(self.txt_guide, guide_text())

    def _build_bottom(self):
        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 8))
        ttk.Label(bar, textvariable=self.var_status, font=self.fonts["small"]
                  ).pack(side="left")
        # Имя автора — постоянно, внизу справа, в обоих языках без перевода.
        ttk.Label(bar, text="SteveWorks001", font=self.fonts["smallb"],
                  foreground="#8a8a82").pack(side="right")

    # ---------- вывод чисел ----------
    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.see("1.0")

    def refresh(self):
        """Пересчитать модель по текущим полям и обновить все панели."""
        self.lrn = Learn(self.state)
        # Шаг 0 = «ещё не начат»; счётчик живёт в var_step, а не в _text_refs,
        # поэтому переводится здесь же.
        self.var_step.set(T("lbl_step", n=self.step, total=TOTAL_STEPS))
        self.lbl_step_name.configure(
            text=T(STEP_BY_N[max(self.step, 1)]["title"]) if self.step else T("c_notyet"),
            foreground={"fwd": COL_FWD, "bwd": COL_BWD, "upd": COL_POS}[
                STEP_BY_N[max(self.step, 1)]["phase"]] if self.step else COL_DIM)
        self._set_text(self.txt_calc, calc_text(self.lrn, self.step))
        self._set_text(self.txt_why, self._why_block())
        self._render_table()
        a, bnd = MODES[self.mode]
        for n, lbl in self.step_dots.items():
            done = n <= self.step
            inside = a <= n <= bnd
            if n == self.step:
                bg, fg = COL_ACTIVE, "#ffffff"
            elif done and inside:
                bg, fg = "#e6efe7", COL_POS
            elif inside:
                bg, fg = "#eceae4", "#6a6a63"
            else:
                bg, fg = COL_BG, "#bcbcb4"
            lbl.configure(bg=bg, fg=fg)
        self.view.redraw()

    def _why_block(self):
        parts = []
        if self.step:
            parts.append(T(STEP_BY_N[self.step]["title"]).upper())
            parts.append("")
            parts.append(why_now(self.lrn, self.step))
        else:
            parts.append(T("c_notyet"))
        if self.step >= 11:
            parts += ["", T("what_hdr"), what_happened(self.lrn)]
        return "\n".join(parts)

    def _render_table(self):
        self.tree.delete(*self.tree.get_children())
        know_out, know_hid = self.step >= 8, self.step >= 9
        for k in WEIGHT_KEYS:
            known = know_out if k in OUTPUT_KEYS else know_hid
            if known:
                self.tree.insert("", "end", iid=k, values=(
                    DISP[k], f4(self.state[k]), f4(self.lrn.grad(k)),
                    f4(self.lrn.delta(k)), f4(self.lrn.new_value(k))))
            else:
                self.tree.insert("", "end", iid=k, values=(
                    DISP[k], f4(self.state[k]), "—", "—", "—"))
        if self.selected in WEIGHT_KEYS:
            try:
                self.tree.selection_set(self.selected)
            except tk.TclError:
                pass
        before = [T("hdr_before"),
                  f"  ŷ = {f4(self.lrn.f['yhat'])}",
                  f"  L = {f4(self.lrn.loss)}"]
        before += [f"  {DISP[k]} = {f4(self.state[k])}" for k in WEIGHT_KEYS]
        if self.step >= 10:
            before += ["", T("hdr_after"),
                       f"  ŷ = {f4(self.lrn.f2['yhat'])}",
                       f"  L = {f4(self.lrn.loss2)}"]
            before += [f"  {DISP[k]} = {f4(self.lrn.new_value(k))}" for k in WEIGHT_KEYS]
        else:
            before += ["", T("c_notyet")]
        self._set_text(self.txt_before, "\n".join(before))

    def _on_select(self, *_):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        if key in WEIGHT_KEYS and key != self.selected:
            self.selected = key
            self.set_status("status_selected", k=DISP[key])
            self.view.redraw()

    # ---------- действия ----------
    def next_step(self):
        a, bnd = MODES[self.mode]
        cur = self.step if a <= self.step <= bnd else a - 1
        if cur >= bnd:
            self.set_status("status_last", n=bnd)
            return
        self.step = max(cur + 1, a)
        self.selected = STEP_FOCUS_PARAM.get(self.step, self.selected)
        self.refresh()
        self.set_status("status_step", n=self.step,
                        title=T(STEP_BY_N[self.step]["title"]))

    def prev_step(self):
        a, bnd = MODES[self.mode]
        if self.step <= a:
            self.set_status("status_first")
            return
        self.step -= 1
        self.selected = STEP_FOCUS_PARAM.get(self.step, self.selected)
        self.refresh()
        self.set_status("status_step", n=self.step,
                        title=T(STEP_BY_N[self.step]["title"]))

    def goto_step(self, n):
        a, bnd = MODES[self.mode]
        if not (a <= n <= bnd):
            self.set_status("status_mode", mode=T("mode_" + self.mode), a=a, b=bnd)
            return
        self.step = n
        self.selected = STEP_FOCUS_PARAM.get(n, self.selected)
        self.refresh()
        self.set_status("status_step", n=n, title=T(STEP_BY_N[n]["title"]))

    def reset(self):
        self.step = 0
        self.refresh()
        self.set_status("status_reset")

    def show_full(self):
        self.mode = "full"
        self.var_mode.set("full")
        self.var_mode_h.set(T("mode_full_hint"))
        self.step = TOTAL_STEPS
        self.selected = "w5"
        self.refresh()
        self.set_status("status_full", n=TOTAL_STEPS)

    def set_mode(self, mode):
        self.mode = mode
        a, bnd = MODES[mode]
        self.var_mode_h.set(T("mode_" + mode + "_hint"))
        self.step = 0
        self.selected = "w5" if mode != "bwd" else "w5"
        self.refresh()
        self.set_status("status_mode", mode=T("mode_" + mode), a=a, b=bnd)

    def apply_example(self, eid):
        ex = next(e for e in EXAMPLES if e["id"] == eid)
        self._silent = True
        self.state = dict(ex["state"])
        for k, fld in self.fields.items():
            fld.set_value(self.state[k])
        self._silent = False
        self.step = 0
        self.refresh()
        self.set_status("status_example", name=T(eid + "_name"))

    def on_param(self, value, raw, key, clamped=False):
        """Поле ввода изменилось: значение либо принято, либо отклонено."""
        if self._silent:
            return
        if value is None:
            self.set_status("status_bad", raw=raw)
            return
        self.state[key] = value
        if clamped:
            lo, hi = LIMITS[key]
            self.set_status("status_range", name=DISP[key], lo=fmt(lo), hi=fmt(hi))
        else:
            self.set_status("status_step", n=self.step,
                            title=T(STEP_BY_N[max(self.step, 1)]["title"])) \
                if self.step else self.set_status("status_ready")
        self.refresh()


def main():
    app = BackpropApp()
    app.mainloop()


if __name__ == "__main__":
    main()

