"""
Измерения на GPU: латентность, пиковая память, энергия + имена CUDA-ядер по слоям.

Файл разбит на ячейки маркерами `# %%` — каждую можно скопировать в отдельную
ячейку ноутбука Colab/Kaggle и запускать по порядку.

Результаты: results/measurements.csv, results/kernels.csv, results/env.json
"""

# %% Ячейка 1. Настройка окружения и флаги из §7 задания
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from models import MyModel

torch.backends.cudnn.benchmark = False        # эвристики PyTorch выбирают ядро
torch.backends.cudnn.allow_tf32 = False       # FP32 остаётся FP32
torch.backends.cuda.matmul.allow_tf32 = False

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Паспортные характеристики карт: (полоса памяти Б/с, FP32 FLOP/с, TDP Вт)
GPU_PEAKS = {
    "T4":   (320.0e9, 8.1e12, 70.0),
}

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
peaks = next((v for k, v in GPU_PEAKS.items() if k in gpu_name), (None, None, None))

model = MyModel().to(device).eval()

env = {
    "gpu": gpu_name,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "total_memory": torch.cuda.get_device_properties(0).total_memory,
    "bw_peak": peaks[0],
    "flops_peak": peaks[1],
    "tdp": peaks[2],
}
print(json.dumps(env, indent=2, ensure_ascii=False))
if peaks[0] is None:
    print("!! карта не найдена в GPU_PEAKS")

# %% Ячейка 2. Энергия через NVML (счётчик в мДж, Volta+). Без него энергия будет NaN
def init_nvml():
    """Модуль и хэндл NVML, либо (None, None), если счётчик энергии недоступен."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # проверяем, что счётчик есть
        return pynvml, handle
    except Exception as exc:                                # noqa: BLE001
        print("NVML недоступен, энергия измеряться не будет:", repr(exc))
        print("в Colab поможет:  !pip install -q nvidia-ml-py")
        return None, None


NVML_LIB, NVML = init_nvml()


def energy_mj():
    """Показание счётчика суммарной энергии GPU, мДж."""
    return NVML_LIB.nvmlDeviceGetTotalEnergyConsumption(NVML) if NVML else float("nan")


# Мощность на простое — нужна как P_idle в энергетической модели
if NVML:
    torch.cuda.synchronize()
    time.sleep(3.0)
    e0, t0 = energy_mj(), time.perf_counter()
    time.sleep(5.0)
    env["p_idle"] = (energy_mj() - e0) / 1e3 / (time.perf_counter() - t0)
    print(f"P_idle = {env['p_idle']:.2f} Вт")
else:
    env["p_idle"] = None


# %% Ячейка 3. Сетка измерений: базовая + случайные точки (они же валидация)
BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 0

rng = random.Random(SEED)
extra_sizes = rng.sample([s for s in range(32, 513, 16) if s not in BASE_SIZES], 4)
extra_batches = rng.sample([b for b in range(1, 257) if b not in BASE_BATCHES], 3)

SIZES = sorted(BASE_SIZES + extra_sizes)
BATCHES = sorted(BASE_BATCHES + extra_batches)

print(f"S ({len(SIZES)}): {SIZES}\nB ({len(BATCHES)}): {BATCHES}\nвсего: {len(SIZES) * len(BATCHES)}")
print(f"случайные S: {sorted(extra_sizes)}, случайные B: {sorted(extra_batches)}")


# %% Ячейка 4. Протокол одного замера
def measure_one(image_size, batch, target_time=0.4, energy_window=0.3, warmup=5):
    """
    Латентность (медиана), пиковая память и энергия одного forward-pass.

    Число повторов подбирается так, чтобы замер занимал ~target_time секунд:
    мелкие конфигурации усредняются по сотням проходов, тяжёлые — по единицам.
    """
    x = torch.randn(batch, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        # грубая оценка, чтобы выбрать число повторов
        t0 = time.perf_counter()
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        est = (time.perf_counter() - t0) / 3

        n_rep = min(201, max(7, int(target_time / est) | 1))
        times = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        # память: reset_peak делает текущее выделение точкой отсчёта, поэтому
        # сам x тоже попадает в пик — как и в аналитической формуле
        torch.cuda.reset_peak_memory_stats()
        model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        joules = float("nan")
        if NVML:
            k = min(2000, max(3, int(energy_window / est)))
            torch.cuda.synchronize()
            e0 = energy_mj()
            for _ in range(k):
                model(x)
            torch.cuda.synchronize()
            joules = (energy_mj() - e0) / 1e3 / k

    del x
    times.sort()
    return {
        "latency": times[len(times) // 2],
        "latency_min": times[0],
        "memory": peak,
        "energy": joules,
        "n_repeats": n_rep,
    }


# %% Ячейка 5. Прогон всей сетки (OOM ловим и записываем как есть)
rows = []
for image_size in SIZES:
    for batch in BATCHES:
        record = {
            "S": image_size,
            "B": batch,
            "is_validation": image_size not in BASE_SIZES or batch not in BASE_BATCHES,
            "oom": False,
        }
        try:
            record.update(measure_one(image_size, batch))
            print(f"S={image_size:>4} B={batch:>3}  "
                  f"{record['latency'] * 1e3:8.3f} мс  "
                  f"{record['memory'] / 1e6:8.1f} МБ  "
                  f"{record['energy'] * 1e3:8.2f} мДж")
        except torch.cuda.OutOfMemoryError:
            record.update(oom=True, latency=None, latency_min=None,
                          memory=None, energy=None, n_repeats=0)
            print(f"S={image_size:>4} B={batch:>3}  OOM")
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rows.append(record)

measurements = pd.DataFrame(rows)
measurements.to_csv(RESULTS / "measurements.csv", index=False)
(RESULTS / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
print(f"\n{len(measurements)} конфигураций, OOM: {int(measurements['oom'].sum())}")


def device_us(evt):
    """Время ядра на устройстве, мкс. Имя атрибута менялось между версиями torch."""
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0.0)


# %% Ячейка 6. Время ядер на устройстве — независимая проверка theta0
# Разность wall-clock и суммарного времени ядер и есть накладные на запуск:
# GPU простаивает, пока CPU ставит ядра в очередь. Меряем отдельным проходом,
# чтобы накладные профайлера не попали в основную латентность из ячейки 5.
def gpu_time_one(image_size, batch, iters=10):
    """Суммарное время ядер GPU на один forward-pass, с."""
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
    del x
    return sum(device_us(e) for e in prof.key_averages()) / iters / 1e6


gpu_times = []
for row in measurements.itertuples():
    if row.oom:
        gpu_times.append(None)
        continue
    try:
        gpu_times.append(gpu_time_one(row.S, row.B))
    except torch.cuda.OutOfMemoryError:
        gpu_times.append(None)
    finally:
        torch.cuda.empty_cache()

measurements["gpu_time"] = gpu_times
measurements["launch_overhead"] = measurements["latency"] - measurements["gpu_time"]
measurements.to_csv(RESULTS / "measurements.csv", index=False)

overhead = measurements["launch_overhead"].dropna()
print(f"накладные на запуск: медиана {overhead.median() * 1e6:.0f} мкс, "
      f"разброс {overhead.min() * 1e6:.0f}..{overhead.max() * 1e6:.0f} мкс")
print("это прямая оценка theta0 — сравни её с тем, что выдаст calibrate.py")


# %% Ячейка 7. Имена CUDA-ядер по слоям
# Каждый слой профилируется отдельно на входе своей формы: так ядро однозначно
# сопоставляется слою. Выбор алгоритма cuDNN зависит от геометрии свёртки,
# а не от соседних слоёв, поэтому изоляция картину не искажает.
KERNEL_SIZES = BASE_SIZES
KERNEL_BATCHES = [1, 8, 64, 256]


def layer_input_shapes(image_size, batch):
    """Форма входа каждого слоя — снимаем хуками за один проход."""
    shapes, handles = {}, []

    def make_hook(layer_name):
        # хук обязан вернуть None: любое другое значение PyTorch подставит
        # вместо выхода слоя, и следующий слой получит его на вход
        def hook(module, inputs, output):
            shapes.setdefault(layer_name, tuple(inputs[0].shape))

        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            model(torch.randn(batch, 3, image_size, image_size, device=device))
    finally:
        for h in handles:
            h.remove()
    return shapes


kernel_rows = []
for image_size in KERNEL_SIZES:
    for batch in KERNEL_BATCHES:
        try:
            shapes = layer_input_shapes(image_size, batch)
            for name, shape in shapes.items():
                module = getattr(model, name)
                inp = torch.randn(*shape, device=device)
                with torch.inference_mode():
                    for _ in range(3):
                        module(inp)
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        module(inp)
                        torch.cuda.synchronize()
                for evt in prof.key_averages():
                    if device_us(evt) > 0:
                        kernel_rows.append({
                            "S": image_size, "B": batch, "layer": name,
                            "kernel": evt.key, "us": device_us(evt),
                        })
                del inp
        except torch.cuda.OutOfMemoryError:
            # Нужно сохранять в выходную таблицу и такие прогоны, в которых произошёл OOM
            print(f"S={image_size} B={batch}: OOM при профилировке, пропускаем")
        finally:
            torch.cuda.empty_cache()

kernels = pd.DataFrame(kernel_rows)
kernels.to_csv(RESULTS / "kernels.csv", index=False)
print(f"{len(kernels)} строк, уникальных ядер: {kernels['kernel'].nunique()}")
print(kernels.groupby("layer")["kernel"].nunique().to_string())


# %% Ячейка 8. Граница OOM: для каждого S ищем максимальный проходящий B
# Отдельные точки за краем сетки мало что дают, поэтому ищем всю границу целиком:
# её можно наложить на кривую Memory(S, B) = ёмкость и сравнить напрямую.
# MEMORY_FRACTION = None — вся память карты (граница уйдёт за край сетки задания).
# MEMORY_FRACTION = 0.25 — искусственный потолок, граница попадёт внутрь сетки,
# зато её положение известно заранее с точностью до байта.
MEMORY_FRACTION = None
FRONTIER_SIZES = SIZES + [768, 1024]
B_LIMIT = 4096

cap = env["total_memory"] * (MEMORY_FRACTION or 1.0)
if MEMORY_FRACTION:
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
print(f"потолок памяти: {cap / 1e9:.2f} ГБ")


def fits(image_size, batch):
    """Проходит ли конфигурация. Возвращает (успех, пик памяти)."""
    try:
        x = torch.randn(batch, 3, image_size, image_size, device=device)
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
        torch.cuda.reset_peak_memory_stats()


def max_batch(image_size):
    """Наибольший проходящий B: удвоение до отказа, затем деление пополам."""
    lo, hi = 0, 1
    while hi <= B_LIMIT and fits(image_size, hi)[0]:
        lo, hi = hi, hi * 2
    if hi > B_LIMIT:
        return lo, None                       # не упёрлись даже на B_LIMIT
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(image_size, mid)[0] else (lo, mid)
    return lo, hi


frontier = []
for image_size in FRONTIER_SIZES:
    b_fit, b_oom = max_batch(image_size)
    peak = fits(image_size, b_fit)[1] if b_fit else None
    # из формулы листа ⑦: B·(104·S² + 3472) + 4.16e6 = cap
    predicted = (cap - 4_161_296) / (104 * image_size ** 2 + 3472)
    frontier.append({
        "S": image_size, "b_max_fits": b_fit, "b_min_oom": b_oom,
        "peak_at_b_max": peak, "b_max_predicted": predicted, "cap": cap,
    })
    print(f"S={image_size:>5}  проходит B={b_fit:>5}  OOM при B={b_oom}  "
          f"пик {(peak or 0) / 1e9:5.2f} ГБ  предсказано B={predicted:8.1f}")

pd.DataFrame(frontier).to_csv(RESULTS / "oom_frontier.csv", index=False)
if MEMORY_FRACTION:
    torch.cuda.set_per_process_memory_fraction(1.0)
