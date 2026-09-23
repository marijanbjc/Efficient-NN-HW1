"""
Калибровка theta по измерениям и проверка на отложенных точках сетки.

Ячейки разделены маркерами `# %%`. Нужны results/measurements.csv и results/env.json.
Результат: results/theta.json

Два решения, общих для всего файла:
  * theta фитится только на базовой сетке; случайные точки (is_validation) в фит
    не входят и служат проверкой на невиданных конфигурациях;
  * невязка берётся в логарифмах — латентность меняется на три порядка, и в
    линейной шкале тяжёлые точки задавили бы всю launch-bound полку.
"""

# %% Ячейка 1. Данные
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import equations as eq

RESULTS = Path("results")
env = json.loads((RESULTS / "env.json").read_text())
data = pd.read_csv(RESULTS / "measurements.csv")

eq.BW_PEAK = env["bw_peak"]          # паспорт той карты, на которой мерили
eq.FLOPS_PEAK = env["flops_peak"]

ok = data[~data["oom"]]
train = ok[~ok["is_validation"]]
val = ok[ok["is_validation"]]


def error(predicted, measured):
    """Средняя относительная ошибка, %."""
    return float(np.abs(predicted / measured - 1).mean() * 100)


print(f"{len(data)} точек, из них OOM {data['oom'].sum()}")
print(f"обучение {len(train)}, валидация {len(val)}")


# %% Ячейка 2. FLOPs и Memory параметров не содержат — просто сверяем с измерениями
for name, df in ("обучение", train), ("валидация", val):
    predicted = eq.memory(df["S"].values, df["B"].values)
    print(f"memory, {name}: ошибка {error(predicted, df['memory'].values):.0f}%")


# %% Ячейка 3. Латентность: theta0, eta_mem, eta_compute
# Сверху eta единицей не ограничиваем — выход за неё это диагностика, а не ошибка:
# eta_compute > 1 означает, что cuDNN выполняет меньше операций, чем мы насчитали,
# eta_mem > 1 — что часть трафика осела в кэше и до DRAM не доехала.
NAMES = ("theta0", "eta_mem", "eta_compute")


def latency_residuals(params, df):
    theta = dict(zip(NAMES, params))
    predicted = eq.latency(df["S"].values, df["B"].values, theta)
    return np.log(predicted) - np.log(df["latency"].values)


# Подгоняем параметры:
# - theta0 (базовое время)
# - eta_mem (эффективность производительности памяти)
# - eta_compute (эффекивность производительности арифметики)
fit = least_squares(
    latency_residuals,                             # Как считаем ошибку
    x0=[train["latency"].min(), 0.6, 0.3],         # Начальные значения параметров
    bounds=([1e-7, 1e-3, 1e-3], [1e-1, 3.0, 3.0]), # В каких диапазонах ищем
    args=(train,),
)
theta = dict(zip(NAMES, fit.x))

print(f"theta0      {theta['theta0'] * 1e6:6.0f} мкс")
print(f"eta_mem     {theta['eta_mem']:6.2f}  = {theta['eta_mem'] * eq.BW_PEAK / 1e9:5.0f} ГБ/с")
print(f"eta_compute {theta['eta_compute']:6.2f}  = {theta['eta_compute'] * eq.FLOPS_PEAK / 1e12:5.2f} ТФЛОП/с")

for name, df in ("обучение", train), ("валидация", val):
    predicted = eq.latency(df["S"].values, df["B"].values, theta)
    print(f"latency, {name}: ошибка {error(predicted, df['latency'].values):.0f}%")

# Какой член выигрывает max в каждой точке. Если memory не побеждает нигде, то
# eta_mem на невязку не влияла и осталась равной начальному приближению —
# это не откалиброванное значение, и в README так и надо написать.
times = [
    np.full(len(ok), theta["theta0"]),
    eq.bytes_moved(ok["S"].values, ok["B"].values) / (theta["eta_mem"] * eq.BW_PEAK),
    eq.flops(ok["S"].values, ok["B"].values) / (theta["eta_compute"] * eq.FLOPS_PEAK),
]
regimes = np.array(["launch", "memory", "compute"])[np.argmax(times, axis=0)]
print("режимы:", dict(zip(*np.unique(regimes, return_counts=True))))


# %% Ячейка 4. Энергия: e_flop и e_byte при измеренной P_idle и фиксированной theta
assert data["energy"].notna().any(), "энергия не измерялась, NVML был недоступен"
P_IDLE = env["p_idle"]


def energy_residuals(params, df):
    theta_energy = {**theta, "p_idle": P_IDLE, "e_flop": params[0], "e_byte": params[1]}
    predicted = eq.energy(df["S"].values, df["B"].values, theta_energy)
    return np.log(predicted) - np.log(df["energy"].values)


# Верхние границы из бюджета мощности: динамическая часть не может стоить больше,
# чем TDP минус простой. Делить бюджет надо на РЕАЛЬНО достигнутые скорости, а не
# на паспортные: карта не выходит на 8.1 ТФЛОП/с, поэтому граница из пикового
# значения оказывается в разы жёстче физической, и фит упирается в неё.
budget = env["tdp"] - P_IDLE
peak_flops = (eq.flops(ok["S"].values, ok["B"].values) / ok["latency"].values).max()
peak_bytes = (eq.bytes_moved(ok["S"].values, ok["B"].values) / ok["latency"].values).max()

# Подгоняем параметры по энергии
fit_energy = least_squares(
    energy_residuals,
    x0=[3e-12, 5e-11],
    bounds=([0.0, 0.0], [budget / peak_flops, budget / peak_bytes]),
    args=(train,),
)
theta_energy = {"p_idle": P_IDLE, "e_flop": fit_energy.x[0], "e_byte": fit_energy.x[1]}

print(f"p_idle {P_IDLE:.1f} Вт (измерено, не подбиралось)")
print(f"e_flop {theta_energy['e_flop'] * 1e12:.2f} пДж/FLOP")
print(f"e_byte {theta_energy['e_byte'] * 1e12:.1f} пДж/байт")
print(f"отношение e_byte/e_flop: {theta_energy['e_byte'] / theta_energy['e_flop']:.0f}")

full = {**theta, **theta_energy}
for name, df in ("обучение", train), ("валидация", val):
    predicted = eq.energy(df["S"].values, df["B"].values, full)
    print(f"energy, {name}: ошибка {error(predicted, df['energy'].values):.0f}%")

power = eq.energy(ok["S"].values, ok["B"].values, full) / eq.latency(ok["S"].values, ok["B"].values, theta)
print(f"предсказанная мощность {power.min():.0f}..{power.max():.0f} Вт при TDP {env['tdp']:.0f}")


# %% Ячейка 5. Сохранение
(RESULTS / "theta.json").write_text(json.dumps({
    "env": env,
    "theta": theta,
    "theta_energy": theta_energy,
}, indent=2, ensure_ascii=False))
print("записано в", RESULTS / "theta.json")
