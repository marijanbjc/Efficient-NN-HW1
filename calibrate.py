"""
Калибровка theta по измерениям и проверка на отложенных точках сетки.

Файл разбит на ячейки маркерами `# %%` — каждую можно скопировать в отдельную
ячейку ноутбука. Требует results/measurements.csv и results/env.json из measure.py.

Результат: results/theta.json
"""

# %% Ячейка 1. Загрузка измерений и паспортных характеристик карты
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import equations as eq   # если переименуешь файл — поменяй импорт здесь

RESULTS = Path("results")
env = json.loads((RESULTS / "env.json").read_text())
data = pd.read_csv(RESULTS / "measurements.csv")

# Паспортные пики берём от той карты, на которой мерили, а не из констант модуля
eq.BW_PEAK = env["bw_peak"]
eq.FLOPS_PEAK = env["flops_peak"]

ok = data[~data["oom"]].copy()
train = ok[~ok["is_validation"]]
val = ok[ok["is_validation"]]
print(f"всего {len(data)}, без OOM {len(ok)}, обучение {len(train)}, валидация {len(val)}")


# %% Ячейка 2. Метрики. Ошибку считаем относительной: латентность меняется
# на три порядка, и в линейной шкале крупные точки задавили бы всю полку
def report(name, pred, meas):
    rel = pred / meas - 1
    print(f"{name:<22} MAPE {np.abs(rel).mean() * 100:6.1f}%   "
          f"медиана {np.median(np.abs(rel)) * 100:6.1f}%   "
          f"макс {np.abs(rel).max() * 100:7.1f}%   "
          f"смещение {rel.mean() * 100:+6.1f}%")
    return np.abs(rel).mean()


# %% Ячейка 3. Функции без параметров — сверяем как есть
print("FLOPs и Memory параметров не содержат, поэтому просто сверяем:\n")
for part, df in (("обучение", train), ("валидация", val)):
    report(f"memory / {part}", eq.memory(df["S"].values, df["B"].values), df["memory"].values)

print("\nПредсказанная память выше измеренной ровно настолько, насколько "
      "PyTorch успевает освобождать тензоры по ходу прохода.")


# %% Ячейка 4. Калибровка latency: theta0, eta_mem, eta_compute
# eta не ограничиваем сверху единицей: выход за неё — это диагностика.
# eta_compute > 1 означает, что cuDNN считает меньше операций, чем мы насчитали
# (Winograd), eta_mem > 1 — что часть трафика осела в кэше и до DRAM не доехала.
LAT_BOUNDS = ([1e-7, 1e-3, 1e-3], [1e-1, 3.0, 3.0])


def lat_residuals(p, df):
    theta = {"theta0": p[0], "eta_mem": p[1], "eta_compute": p[2]}
    return np.log(eq.latency(df["S"].values, df["B"].values, theta)) - np.log(df["latency"].values)


start = [train["latency"].min(), 0.6, 0.3]
fit = least_squares(lat_residuals, start, bounds=LAT_BOUNDS, args=(train,))
theta = {"theta0": fit.x[0], "eta_mem": fit.x[1], "eta_compute": fit.x[2]}

print(f"theta0      = {theta['theta0'] * 1e6:8.1f} мкс")
print(f"eta_mem     = {theta['eta_mem']:8.3f}  ->  {theta['eta_mem'] * eq.BW_PEAK / 1e9:6.1f} ГБ/с")
print(f"eta_compute = {theta['eta_compute']:8.3f}  ->  {theta['eta_compute'] * eq.FLOPS_PEAK / 1e12:6.2f} ТФЛОП/с")
print(f"излом roofline: {theta['eta_compute'] * eq.FLOPS_PEAK / (theta['eta_mem'] * eq.BW_PEAK):.1f} FLOP/байт "
      f"(паспортный {eq.FLOPS_PEAK / eq.BW_PEAK:.1f})\n")

# Какой член выигрывает max в каждой точке. Если memory-член не выигрывает нигде,
# eta_mem на невязку не влияет и останется равной начальному приближению — тогда
# это не откалиброванное значение, а заглушка, и в README так и надо написать.
t_launch = np.full(len(ok), theta["theta0"])
t_mem = eq.bytes_moved(ok["S"].values, ok["B"].values) / (theta["eta_mem"] * eq.BW_PEAK)
t_cmp = eq.flops(ok["S"].values, ok["B"].values) / (theta["eta_compute"] * eq.FLOPS_PEAK)
winner = np.array(["launch", "memory", "compute"])[np.argmax([t_launch, t_mem, t_cmp], axis=0)]
print("режимы по сетке:", dict(zip(*np.unique(winner, return_counts=True))))
if "memory" not in winner:
    print("!! memory-член не доминирует ни в одной точке — eta_mem не определяется этими данными\n")
else:
    print()

lat_train = report("latency / обучение",
                   eq.latency(train["S"].values, train["B"].values, theta), train["latency"].values)
lat_val = report("latency / валидация",
                 eq.latency(val["S"].values, val["B"].values, theta), val["latency"].values)


# %% Ячейка 5. Калибровка energy: e_flop, e_byte при измеренной P_idle и фикс. theta
e_train = train[train["energy"].notna()]
e_val = val[val["energy"].notna()]

p_idle = env.get("p_idle")
if p_idle is None or not len(e_train):
    print("энергия не измерялась — пропускаем калибровку")
    theta_energy = None
else:
    def energy_residuals(p, df):
        te = {**theta, "p_idle": p_idle, "e_flop": p[0], "e_byte": p[1]}
        return np.log(eq.energy(df["S"].values, df["B"].values, te)) - np.log(df["energy"].values)

    # верхние границы из бюджета мощности: вся динамика не может превысить TDP
    dyn = env["tdp"] - p_idle
    bounds = ([0.0, 0.0], [dyn / eq.FLOPS_PEAK, dyn / eq.BW_PEAK])
    fit_e = least_squares(energy_residuals, [1e-12, 5e-11], bounds=bounds, args=(e_train,))
    theta_energy = {"p_idle": p_idle, "e_flop": fit_e.x[0], "e_byte": fit_e.x[1]}

    print(f"P_idle = {p_idle:.2f} Вт (измерено)")
    print(f"e_flop = {theta_energy['e_flop'] * 1e12:.3f} пДж/FLOP   "
          f"(верхняя граница {bounds[1][0] * 1e12:.2f})")
    print(f"e_byte = {theta_energy['e_byte'] * 1e12:.1f} пДж/байт    "
          f"(верхняя граница {bounds[1][1] * 1e12:.1f})")
    print(f"отношение e_byte/e_flop = {theta_energy['e_byte'] / theta_energy['e_flop']:.0f}\n")

    te = {**theta, **theta_energy}
    report("energy / обучение",
           eq.energy(e_train["S"].values, e_train["B"].values, te), e_train["energy"].values)
    report("energy / валидация",
           eq.energy(e_val["S"].values, e_val["B"].values, te), e_val["energy"].values)

    power = (eq.energy(ok["S"].values, ok["B"].values, te)
             / eq.latency(ok["S"].values, ok["B"].values, theta))
    print(f"\nсредняя мощность по сетке: {power.min():.1f}..{power.max():.1f} Вт при TDP {env['tdp']} Вт")


# %% Ячейка 6. Сохранение
out = {
    "env": env,
    "theta": theta,
    "theta_energy": theta_energy,
    "fit": {"latency_mape_train": lat_train, "latency_mape_val": lat_val},
}
(RESULTS / "theta.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
