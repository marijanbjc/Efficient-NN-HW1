"""
Аналитическая модель стоимости сети из models.py. Вывод — в hw1_handwritten.pdf.

    FLOPs(S, B)  = B · (17712·S² + 313344)
    Memory(S, B) = B · (104·S² + 3472) + 4.16·10⁶
    Bytes(S, B)  = B · (196·S² + 8592) + 4.16·10⁶
    T(S, B, θ)   = max(θ₀, Bytes/(η_mem·BW_peak), FLOPs/(η_compute·FLOPs_peak))
    E(S, B, θ)   = P_idle·T + e_flop·FLOPs + e_byte·Bytes

S — сторона квадратного входного изображения (кратна 16), B — размер батча.
Аргументы могут быть массивами numpy: поверхность над плоскостью (S, B)
считается одним вызовом.
"""

from typing import Mapping, Union

import numpy as np

Number = Union[int, float, np.ndarray]

# Коэффициенты при S² и свободные члены — на один сэмпл (листы 4, 7, 8).
FLOPS_S2 = 17_712         # 2 · Σ C_out·C_in·k²/r² по шести свёрткам
FLOPS_CONST = 313_344     # 2 · (512·256 + 256·100); от S не зависит — перед головой GAP
MEM_S2 = 104              # 4 байта · (3+8+2+4+2+4+1+2) элементов активаций
MEM_CONST = 3_472         # 4 · (512 + 256 + 100)
BYTES_S2 = 196            # 4 · Σ(прочитано + записано) по слоям = 4 · 49
BYTES_CONST = 8_592       # 4 · 2148
PARAMS_BYTES = 4_161_296  # 4 · 1 040 324 параметра

# Паспорт ускорителя — ЗАМЕНИТЬ на карту, где снимались измерения (сейчас Tesla T4).
BW_PEAK = 320.0e9         # байт/с
FLOPS_PEAK = 8.1e12       # FLOP/с, FP32


def flops(image_size: Number, batch: Number) -> Number:
    """Число операций с плавающей точкой за один forward-pass."""
    s2 = np.asarray(image_size, float) ** 2
    b = np.asarray(batch, float)
    return b * (FLOPS_S2 * s2 + FLOPS_CONST)


def memory(image_size: Number, batch: Number) -> Number:
    """Память под активации и веса за один forward-pass, байт."""
    s2 = np.asarray(image_size, float) ** 2
    b = np.asarray(batch, float)
    return b * (MEM_S2 * s2 + MEM_CONST) + PARAMS_BYTES


def bytes_moved(image_size: Number, batch: Number) -> Number:
    """Трафик между памятью и чипом за один forward-pass, байт."""
    s2 = np.asarray(image_size, float) ** 2
    b = np.asarray(batch, float)
    return b * (BYTES_S2 * s2 + BYTES_CONST) + PARAMS_BYTES


def latency(image_size: Number, batch: Number, theta: Mapping[str, float]) -> Number:
    """
    Время одного forward-pass, с.

    theta: theta0 — накладные на запуск ядер, с;
    eta_mem и eta_compute — достижимые доли паспортной полосы памяти и паспортной производительности.
    """
    return np.maximum.reduce(np.broadcast_arrays(
        theta["theta0"],
        bytes_moved(image_size, batch) / (theta["eta_mem"] * BW_PEAK),
        flops(image_size, batch) / (theta["eta_compute"] * FLOPS_PEAK),
    ))


def energy(image_size: Number, batch: Number, theta_energy: Mapping[str, float]) -> Number:
    """
    Энергия одного forward-pass, Дж.

    theta_energy: p_idle — мощность на простое, Вт;
    e_flop — Дж на операцию;
    e_byte — Дж на перемещённый байт.
    Плюс ключи theta для latency(): статическая часть зависит от времени прохода,
    поэтому сначала калибруется theta.
    """
    return (
        theta_energy["p_idle"] * latency(image_size, batch, theta_energy)
        + theta_energy["e_flop"] * flops(image_size, batch)
        + theta_energy["e_byte"] * bytes_moved(image_size, batch)
    )
