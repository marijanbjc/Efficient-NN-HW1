"""
Измерения на GPU для ДЗ-1: латентность, пиковая память, энергия, имена CUDA-ядер.

Результаты: results/{measurements.csv, oom_frontier.csv, env.json}
"""

# %% Ячейка 1. Окружение и флаги из §7 задания
import json
import random
import time
from pathlib import Path
from statistics import median

import pandas as pd
import torch

from models import MyModel

torch.backends.cudnn.benchmark = False        # ядро выбирают эвристики, а не автотюнер
torch.backends.cudnn.allow_tf32 = False       # FP32 остаётся FP32
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = torch.device("cuda")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Паспортные характеристики: (полоса памяти Б/с, FP32 FLOP/с, TDP Вт)
GPU_PEAKS = {
    "T4": (320.0e9, 8.1e12, 70.0),
}

model = MyModel().to(DEVICE).eval()
gpu = torch.cuda.get_device_name(0)
peaks = [v for k, v in GPU_PEAKS.items() if k in gpu]
assert peaks, f"{gpu} нет в GPU_PEAKS — впиши паспортные значения этой карты"
bw_peak, flops_peak, tdp = peaks[0]

env = {
    "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "total_memory": torch.cuda.get_device_properties(0).total_memory,
    "bw_peak": bw_peak, "flops_peak": flops_peak, "tdp": tdp,
}
print(json.dumps(env, indent=2, ensure_ascii=False))


# %% Ячейка 2. Энергия: счётчик NVML в мДж (Volta+). P_idle — статический член модели
def init_nvml():
    """Модуль и хэндл NVML либо (None, None). try/except внутри функции, чтобы
    при копировании в ячейку ветка ошибки не отклеилась и не выполнилась сама."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return pynvml, handle
    except Exception as exc:                                # noqa: BLE001
        print("NVML недоступен, энергия будет NaN:", repr(exc))
        print("в Colab поможет:  !pip install -q nvidia-ml-py")
        return None, None


NVML_LIB, NVML = init_nvml()


def energy_j():
    """Показание счётчика суммарной энергии GPU, Дж."""
    return NVML_LIB.nvmlDeviceGetTotalEnergyConsumption(NVML) / 1e3 if NVML else float("nan")


if NVML:
    torch.cuda.synchronize()
    time.sleep(10.0)
    e0, t0 = energy_j(), time.perf_counter()
    time.sleep(15.0)
    env["p_idle"] = (energy_j() - e0) / (time.perf_counter() - t0)
    print(f"P_idle = {env['p_idle']:.2f} Вт")
else:
    env["p_idle"] = None


# %% Ячейка 3. Сетка: базовая из задания + случайные точки, они же валидация
BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 0

rng = random.Random(SEED)
SIZES = sorted(BASE_S + rng.sample([s for s in range(32, 513, 16) if s not in BASE_S], 4))
BATCHES = sorted(BASE_B + rng.sample([b for b in range(1, 257) if b not in BASE_B], 3))

print(f"S ({len(SIZES)}): {SIZES}")
print(f"B ({len(BATCHES)}): {BATCHES}")
print(f"конфигураций: {len(SIZES) * len(BATCHES)}")


# %% Ячейка 4. Протокол одного замера
def measure_one(image_size, batch, t_lat=0.4, t_energy=1.5, warmup=5):
    """
    Возвращает латентность, время на устройстве, пик памяти, энергию и мощность.

    Допущения протокола:
      * прогрев 5 проходов — убираем ленивую инициализацию cuDNN и разгон частот;
      * latency — медиана wall-clock с synchronize после каждого прохода (конвенция §5);
      * device_time — то же самое по CUDA events, то есть без времени на стороне CPU:
        разность latency - device_time даёт прямую оценку накладных на запуск;
      * memory — max_memory_allocated на отдельном проходе. Входной x выделен до
        reset_peak, поэтому остаётся в пике — как и в аналитической формуле;
      * energy — дельта счётчика NVML за k проходов подряд, делённая на k. Внутри
        окна synchronize нет, проходы конвейеризуются и идут быстрее медианной
        латентности, поэтому мощность считаем по собственному времени окна.
    """
    x = torch.randn(batch, 3, image_size, image_size, device=DEVICE)
    ev_start, ev_stop = torch.cuda.Event(True), torch.cuda.Event(True)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        # число повторов подбираем под t_lat: мелкие точки усредняем по сотням
        # проходов, тяжёлые — по единицам, иначе прогон растянется на часы
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        est = time.perf_counter() - t0
        n_rep = min(201, max(7, int(t_lat / est)))

        wall, dev = [], []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            ev_start.record()
            model(x)
            ev_stop.record()
            torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
            dev.append(ev_start.elapsed_time(ev_stop) / 1e3)

        torch.cuda.reset_peak_memory_stats()
        model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        energy = power = float("nan")
        if NVML:
            k = min(4000, max(3, int(t_energy / est)))
            torch.cuda.synchronize()
            e0, t0 = energy_j(), time.perf_counter()
            for _ in range(k):
                model(x)
            torch.cuda.synchronize()
            window = time.perf_counter() - t0
            energy = (energy_j() - e0) / k
            power = energy * k / window

    return {"latency": median(wall), "device_time": median(dev),
            "memory": peak, "energy": energy, "power": power}


# %% Ячейка 5. Прогон сетки -> measurements.csv
rows = []
for image_size in SIZES:
    for batch in BATCHES:
        row = {
            "S": image_size,
            "B": batch,
            "is_validation": image_size not in BASE_S or batch not in BASE_B,
            "oom": False,
        }
        try:
            row.update(measure_one(image_size, batch))
            print(f"S={image_size:>4} B={batch:>3}  {row['latency'] * 1e3:8.3f} мс  "
                  f"{row['memory'] / 1e6:8.1f} МБ  {row['energy'] * 1e3:8.2f} мДж  "
                  f"{row['power']:5.1f} Вт")
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True          # остальные поля останутся пустыми -> NaN в csv
            print(f"S={image_size:>4} B={batch:>3}  OOM")
        finally:
            torch.cuda.empty_cache()
        rows.append(row)

pd.DataFrame(rows).to_csv(RESULTS / "measurements.csv", index=False)
(RESULTS / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
print(f"\n{len(rows)} конфигураций, OOM: {sum(r['oom'] for r in rows)}")


# %% Ячейка 6. Граница OOM -> oom_frontier.csv
# На полной памяти край сетки задания (B=256) до OOM не доходит, поэтому ищем
# саму границу: максимальный проходящий B для каждого S. Её можно наложить на
# кривую Memory(S, B) = ёмкость и сравнить с предсказанием напрямую.
# MEMORY_FRACTION = 0.25 опускает потолок так, что граница попадает внутрь сетки,
# зато ёмкость тогда известна точно, а не за вычетом контекста драйвера.
MEMORY_FRACTION = None
FRONTIER_S = SIZES + [768, 1024]
B_LIMIT = 4096

cap = env["total_memory"] * (MEMORY_FRACTION or 1.0)
if MEMORY_FRACTION:
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)


def fits(image_size, batch):
    """Проходит ли конфигурация; вторым значением — пик памяти."""
    try:
        x = torch.randn(batch, 3, image_size, image_size, device=DEVICE)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del x
        return True, peak
    except torch.cuda.OutOfMemoryError:
        return False, None
    finally:
        torch.cuda.empty_cache()


frontier = []
for image_size in FRONTIER_S:
    lo, hi = 0, 1                                   # удвоение до первого отказа
    while hi <= B_LIMIT and fits(image_size, hi)[0]:
        lo, hi = hi, hi * 2
    if hi > B_LIMIT:
        b_fit, b_oom = lo, None
    else:
        while hi - lo > 1:                          # затем деление пополам
            mid = (lo + hi) // 2
            if fits(image_size, mid)[0]:
                lo = mid
            else:
                hi = mid
        b_fit, b_oom = lo, hi
    peak = fits(image_size, b_fit)[1] if b_fit else None
    predicted = (cap - 4_161_296) / (104 * image_size ** 2 + 3472)   # из формулы Memory
    frontier.append({"S": image_size, "b_max_fits": b_fit, "b_min_oom": b_oom,
                     "peak_at_b_max": peak, "b_max_predicted": predicted, "cap": cap})
    print(f"S={image_size:>5}  проходит B={b_fit:>5}  OOM при B={b_oom}  "
          f"пик {(peak or 0) / 1e9:5.2f} ГБ  предсказано B={predicted:8.1f}")

pd.DataFrame(frontier).to_csv(RESULTS / "oom_frontier.csv", index=False)
if MEMORY_FRACTION:
    torch.cuda.set_per_process_memory_fraction(1.0)

# %% Ячейка 7. Достижимая полоса памяти -> eta_mem
# Сеть нигде не упирается в память (её интенсивность ~90 FLOP/байт против ~13 у
# карты), поэтому из основного прогона eta_mem не определяется — фит его просто
# не трогает. Меряем отдельно, как и P_idle: копирование большого тензора читает
# и пишет ровно свой размер и не делает ни одной арифметической операции.
def measure_bandwidth(size_bytes=512 * 2 ** 20, repeats=20):
    """Достижимая полоса памяти, байт/с."""
    src = torch.empty(size_bytes // 4, dtype=torch.float32, device=DEVICE)
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        dst.copy_(src)
    torch.cuda.synchronize()
    return 2 * size_bytes * repeats / (time.perf_counter() - t0)   # чтение + запись


env["bw_achieved"] = measure_bandwidth()
env["eta_mem"] = env["bw_achieved"] / env["bw_peak"]
(RESULTS / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
print(f"полоса {env['bw_achieved'] / 1e9:.0f} ГБ/с из паспортных {env['bw_peak'] / 1e9:.0f}")
print(f"eta_mem = {env['eta_mem']:.2f}")


# %% Ячейка 8. Прогон с искусственным потолком памяти -> measurements_capped.csv
# На полной памяти сетка нигде не переполняется, поэтому проверить формулу Memory
# на границе не на чем. Опускаем потолок до 10% памяти карты: тогда часть
# конфигураций перестаёт влезать, а сама ёмкость известна точно — и предсказание
# "упадёт там, где Memory(S, B) > потолка" становится проверяемым утверждением.
#
# Здесь нужен только факт "влезло или нет", поэтому делаем один проход на
# конфигурацию вместо полного замера — это секунды вместо десяти минут.
CAP_FRACTION = 0.10

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(CAP_FRACTION)
cap_bytes = env["total_memory"] * CAP_FRACTION
print(f"потолок памяти {cap_bytes / 1e9:.2f} ГБ")

capped_rows = []
for image_size in SIZES:
    for batch in BATCHES:
        row = {"S": image_size, "B": batch, "cap": cap_bytes, "oom": False}
        try:
            x = torch.randn(batch, 3, image_size, image_size, device=DEVICE)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                model(x)
            torch.cuda.synchronize()
            row["memory"] = torch.cuda.max_memory_allocated()
            del x
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True
        finally:
            torch.cuda.empty_cache()
        capped_rows.append(row)

torch.cuda.set_per_process_memory_fraction(1.0)

capped = pd.DataFrame(capped_rows)
capped.to_csv(RESULTS / "measurements_capped.csv", index=False)
print(f"переполнений: {capped['oom'].sum()} из {len(capped)}")
