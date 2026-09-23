"""
Аналитическая модель стоимости свёрточной сети из models.py.

Все функции принимают скаляры и numpy-массивы (broadcasting): поверхность над
плоскостью (S, B) считается одним вызовом. Всюду S — сторона квадратного
входного изображения (кратна 16), B — размер батча.

Соглашения (вывод — в hw1_handwritten.pdf):
  * FP32, 4 байта на элемент, eval() + inference_mode().
  * 1 MAC = 2 FLOPs, считаются только Conv2d и Linear.
  * Memory — сумма всех активаций одновременно плюс веса (упрощение,
    согласованное с преподавателем); реальный пик вдвое ниже, см. memory_peak().
  * Bytes moved — трафик DRAM <-> чип: каждый слой читает входы
    один раз и пишет выход один раз; веса читаются раз за проход, от B не зависят.
  * Latency — глобальная задержка: max(накладные, память, арифметика).
  * Energy — статика за время прохода плюс динамика по FLOPs и по байтам.
"""

from typing import Mapping, Union

import numpy as np

Number = Union[int, float, np.ndarray]

# Коэффициенты при S² и свободные члены — на один сэмпл.
FLOPS_S2 = 17_712         # 2 * Σ C_out·C_in·k²/r² по шести свёрткам
FLOPS_HEAD = 313_344      # 2 * (512·256 + 256·100); от S не зависит — перед головой GAP
MEM_S2 = 104              # 4 байта * (3 + 8 + 2 + 4 + 2 + 4 + 1 + 2) элементов активаций
MEM_HEAD = 3_472          # 4 * (512 + 256 + 100)
BYTES_S2 = 196            # 4 * Σ(чтение + запись) по ядрам = 4 * 49; с ReLU между свёртками было бы 364
BYTES_HEAD = 8_592        # 4 * 2148
PARAMS_BYTES = 4_161_296  # 4 * 1_040_324 параметра; в bytes_moved это же число — веса читаются раз за проход

# Пик по времени жизни тензоров: на MaxPool одновременно живы вход сети (12·S²),
# выход conv1 (32·S²) и выход пулинга (8·S²).
MEM_PEAK_S2 = 52

# Паспорт ускорителя — ЗАМЕНИТЬ на карту, где снимались измерения (сейчас Tesla T4).
BW_PEAK = 320.0e9         # байт/с
FLOPS_PEAK = 8.1e12       # FLOP/с, FP32 без тензорных ядер


def flops(image_size: Number, batch: Number) -> Number:
    """Количество операций с плавающей точкой на один forward-pass."""
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * (FLOPS_S2 * s ** 2 + FLOPS_HEAD)


def memory(image_size: Number, batch: Number) -> Number:
    """Занятая память на один forward-pass, байт. Без workspace cuDNN — нижняя оценка."""
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * (MEM_S2 * s ** 2 + MEM_HEAD) + PARAMS_BYTES


def memory_peak(image_size: Number, batch: Number) -> Number:
    """Пик памяти с учётом освобождения тензоров по ходу прохода, байт. Для §5."""
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * MEM_PEAK_S2 * s ** 2 + PARAMS_BYTES


def bytes_moved(image_size: Number, batch: Number) -> Number:
    """Трафик между DRAM и чипом за один forward-pass, байт."""
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return b * (BYTES_S2 * s ** 2 + BYTES_HEAD) + PARAMS_BYTES


def _times(image_size: Number, batch: Number, theta: Mapping[str, float]) -> list:
    """Три конкурирующих времени: накладные, память, арифметика."""
    bw = theta.get("bw_peak", BW_PEAK)
    fp = theta.get("flops_peak", FLOPS_PEAK)
    return np.broadcast_arrays(
        theta["theta0"],
        bytes_moved(image_size, batch) / (theta["eta_mem"] * bw),
        flops(image_size, batch) / (theta["eta_compute"] * fp),
    )


def latency(image_size: Number, batch: Number, theta: Mapping[str, float]) -> Number:
    """
    Время одного forward-pass, с.

    theta: theta0 [с] — накладные на запуск ядер прохода; eta_mem и eta_compute —
    достижимые доли пиковой полосы и пиковой арифметики. Необязательные ключи
    bw_peak / flops_peak переопределяют паспортные константы модуля.
    """
    return np.maximum.reduce(_times(image_size, batch, theta))


def energy(image_size: Number, batch: Number, theta_energy: Mapping[str, object]) -> Number:
    """
    Энергия одного forward-pass, Дж.

    theta_energy: p_idle [Вт], e_flop [Дж/FLOP], e_byte [Дж/байт] и theta_latency —
    параметры latency(), поскольку статический член зависит от времени. Отсюда
    порядок калибровки: сначала theta, затем theta_energy при фиксированной theta.
    """
    return (
        theta_energy["p_idle"] * latency(image_size, batch, theta_energy["theta_latency"])
        + theta_energy["e_flop"] * flops(image_size, batch)
        + theta_energy["e_byte"] * bytes_moved(image_size, batch)
    )


def arithmetic_intensity(image_size: Number, batch: Number) -> Number:
    """FLOPs на байт трафика. Выше точки излома roofline сеть compute-bound."""
    return flops(image_size, batch) / bytes_moved(image_size, batch)


def mean_power(image_size: Number, batch: Number, theta_energy: Mapping[str, object]) -> Number:
    """Средняя мощность за проход, Вт."""
    t = latency(image_size, batch, theta_energy["theta_latency"])
    return energy(image_size, batch, theta_energy) / t


def regime(image_size: Number, batch: Number, theta: Mapping[str, float]) -> np.ndarray:
    """Что ограничивает проход в точке сетки: "launch" / "memory" / "compute"."""
    times = _times(image_size, batch, theta)
    return np.array(["launch", "memory", "compute"])[np.argmax(times, axis=0)]
