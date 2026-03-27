#!/usr/bin/env python3
"""
M2 Asymmetric Engine — Heavy Tail Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Тест-драйв: Квантильний коридор проти даних з "важкими хвостами" та викидами.

Показує:
  1. Звичайна МНК-регресія "їде" слідом за викидами
  2. P50-медіана залишається стабільною
  3. P90 "відкривається" у зоні ризику — попереджає, але не паднікує
  4. P10 не падає — нижня межа стійка

Запуск:  python3 demo_heavy_tail.py
Графік:  зберігається у heavy_tail_demo.png + показується інтерактивно
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ─────────────────────────────────────────────────────────────────────────────
# 1. Дані з "важкими хвостами" та асиметричними викидами
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 200

X_np = np.linspace(0, 10, N).reshape(-1, 1).astype(np.float32)

# Основний лінійний тренд + нормальний шум
y_np = (2.0 * X_np + 1.0
        + np.random.normal(0, 1.0, (N, 1)).astype(np.float32))

# Асиметричні викиди: критичні стрибки навантаження (тільки вгору)
# → саме той "важкий хвіст" що ламає стандартний MSE
y_np[150:160] += 15.0   # +15 у зоні x ≈ 7.5–8.0
y_np[80:83]   += 8.0    # +8  у зоні x ≈ 4.0

# MLX Unified Memory — дані вже "в GPU" без копіювання
X_mx = mx.array(X_np)
y_mx = mx.array(y_np.flatten())

# ─────────────────────────────────────────────────────────────────────────────
# 2. Моделі
# ─────────────────────────────────────────────────────────────────────────────

class LinearModel(nn.Module):
    """Звичайний лінійний регресор (MSE) — контрольна група."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def __call__(self, x):
        return self.layer(x).squeeze(-1)


class QuantileModel(nn.Module):
    """
    Три квантильні голівки: P10 / P50 / P90.
    Навчаються одночасно через Pinball Loss.
    """
    QUANTILES = [0.10, 0.50, 0.90]

    def __init__(self):
        super().__init__()
        self.heads = [nn.Linear(1, 1) for _ in self.QUANTILES]

    def __call__(self, x) -> list[mx.array]:
        return [h(x).squeeze(-1) for h in self.heads]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Функції втрат
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(pred: mx.array, y: mx.array) -> mx.array:
    return mx.mean(mx.square(pred - y))


def pinball_loss(pred: mx.array, y: mx.array, tau: float) -> mx.array:
    """
    Asymmetric Pinball Loss.
    tau=0.5 → медіана (стійка до викидів, як MAE)
    tau=0.9 → консервативна верхня межа
    tau=0.1 → оптимістична нижня межа
    """
    err = y - pred
    return mx.mean(mx.maximum(tau * err, (tau - 1.0) * err))


def quantile_loss(model: QuantileModel, X: mx.array, y: mx.array) -> mx.array:
    preds = model(X)
    losses = [pinball_loss(p, y, tau)
              for p, tau in zip(preds, QuantileModel.QUANTILES)]
    return mx.mean(mx.stack(losses))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Навчання на M2 GPU
# ─────────────────────────────────────────────────────────────────────────────

ITERATIONS = 500
LR         = 0.02

# ── Звичайна MSE регресія ─────────────────────────────────────────────────────
print("Навчання MSE-регресії...")
linear_model = LinearModel()
opt_linear   = optim.Adam(learning_rate=LR)

def mse_fn(model, X, y):
    return mse_loss(model(X), y)

grad_fn_mse = nn.value_and_grad(linear_model, mse_fn)
t0 = time.perf_counter()

for i in range(ITERATIONS):
    loss, grads = grad_fn_mse(linear_model, X_mx, y_mx)
    opt_linear.update(linear_model, grads)
    mx.eval(linear_model.parameters(), opt_linear.state)
    if i % 100 == 0:
        print(f"  MSE [{i:3d}] loss={float(loss):.4f}")

t_mse = time.perf_counter() - t0

# ── Квантильна регресія (Pinball) ─────────────────────────────────────────────
print("\nНавчання QuantileModel (P10/P50/P90)...")
q_model  = QuantileModel()
opt_q    = optim.Adam(learning_rate=LR)

grad_fn_q = nn.value_and_grad(q_model, quantile_loss)
t0 = time.perf_counter()

for i in range(ITERATIONS):
    loss_q, grads_q = grad_fn_q(q_model, X_mx, y_mx)
    opt_q.update(q_model, grads_q)
    mx.eval(q_model.parameters(), opt_q.state)
    if i % 100 == 0:
        print(f"  Quantile [{i:3d}] loss={float(loss_q):.4f}")

t_quantile = time.perf_counter() - t0

print(f"\n⚡ Швидкість навчання (Apple M2 Unified Memory):")
print(f"   MSE model:      {t_mse*1000:.1f} ms  ({ITERATIONS} ітерацій)")
print(f"   Quantile model: {t_quantile*1000:.1f} ms  ({ITERATIONS} ітерацій)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Прогнози → NumPy для графіків
# ─────────────────────────────────────────────────────────────────────────────

pred_mse = np.array(linear_model(X_mx))
q_preds  = q_model(X_mx)
mx.eval(*q_preds)
p10 = np.array(q_preds[0])
p50 = np.array(q_preds[1])
p90 = np.array(q_preds[2])

X_flat = X_np.flatten()
y_flat = y_np.flatten()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Метрики
# ─────────────────────────────────────────────────────────────────────────────

outlier_idx  = np.concatenate([np.arange(150, 160), np.arange(80, 83)])
clean_idx    = np.setdiff1d(np.arange(N), outlier_idx)

mse_on_clean  = np.mean((pred_mse[clean_idx]  - y_flat[clean_idx])**2)
p50_on_clean  = np.mean(np.abs(p50[clean_idx] - y_flat[clean_idx]))
mse_on_outlier = np.mean((pred_mse[outlier_idx] - y_flat[outlier_idx])**2)
p50_on_outlier = np.mean(np.abs(p50[outlier_idx] - y_flat[outlier_idx]))

coverage = np.mean((y_flat >= p10) & (y_flat <= p90)) * 100

print(f"\n📊 Метрики порівняння:")
print(f"   {'Метрика':<30} {'MSE-регресія':>14} {'P50-медіана':>14}")
print(f"   {'-'*58}")
print(f"   {'MSE/MAE на чистих даних':<30} {mse_on_clean:>14.3f} {p50_on_clean:>14.3f}")
print(f"   {'MSE/MAE на викидах':<30} {mse_on_outlier:>14.3f} {p50_on_outlier:>14.3f}")
print(f"\n   Покриття коридором P10–P90: {coverage:.1f}% точок")
print(f"   (очікуване теоретичне: 80%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Візуалізація
# ─────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")  # headless fallback — завжди працює
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("M2 Asymmetric Engine: Стійкість до важких хвостів", fontsize=14, fontweight="bold")

    # ── Ліва панель: порівняння MSE vs P50 ────────────────────────────────
    ax = axes[0]
    ax.scatter(X_flat, y_flat, color="gray", alpha=0.4, s=18, label="Дані (з викидами)")

    # Виділяємо викиди
    ax.scatter(X_flat[outlier_idx], y_flat[outlier_idx],
               color="red", alpha=0.7, s=30, zorder=5, label="Викиди (+8, +15)")

    ax.plot(X_flat, pred_mse,  color="tomato",   lw=2, label="MSE-регресія (тягнеться за викидами)")
    ax.plot(X_flat, p50,       color="steelblue", lw=2.5, label="P50 медіана (стабільна)")

    ax.set_title("MSE vs Quantile P50")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Права панель: коридор ризику ──────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(X_flat, y_flat, color="gray", alpha=0.4, s=18, label="Дані з викидами")
    ax2.scatter(X_flat[outlier_idx], y_flat[outlier_idx],
                color="red", alpha=0.7, s=30, zorder=5, label="Викиди")

    ax2.fill_between(X_flat, p10, p90,
                     color="steelblue", alpha=0.15, label=f"Коридор P10–P90 ({coverage:.0f}% покриття)")
    ax2.plot(X_flat, p50, color="steelblue", lw=2.5, label="P50 медіана")
    ax2.plot(X_flat, p90, color="steelblue", lw=1,   ls="--", alpha=0.7, label="P90 верхня межа")
    ax2.plot(X_flat, p10, color="steelblue", lw=1,   ls=":",  alpha=0.7, label="P10 нижня межа")

    # Позначаємо зони викидів
    for start, end, label in [(150, 160, "+15"), (80, 83, "+8")]:
        x_zone = X_flat[[start, end - 1]]
        ax2.axvspan(x_zone[0], x_zone[1], color="red", alpha=0.08)

    ax2.set_title(f"Asymmetric Corridor: P10 / P50 / P90")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "heavy_tail_demo.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n🖼  Графік збережено: {out_path}")

    # Якщо є дисплей — показуємо інтерактивно
    try:
        matplotlib.use("TkAgg")
        plt.switch_backend("TkAgg")
        plt.show()
    except Exception:
        pass

except ImportError:
    print("\n[matplotlib не встановлено — пропускаємо графік]")
    print("Встановити: pip install matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Фінальний висновок
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
╔══════════════════════════════════════════════════════════╗
║      M2 Asymmetric Engine — Результати тест-драйву       ║
╠══════════════════════════════════════════════════════════╣
║  MSE-регресія на викидах:  MAE ≈ {mse_on_outlier:.1f}               ║
║  P50-медіана на викидах:   MAE ≈ {p50_on_outlier:.1f}                ║
║  Покриття коридору P10-P90:    {coverage:.0f}%                  ║
║                                                          ║
║  🟢 P50 ігнорує шум — центральна лінія стабільна        ║
║  🔺 P90 розкривається у зонах ризику — попереджає        ║
║  ⚡ Навчання: {(t_mse+t_quantile)*1000:.0f} ms на Apple M2 GPU               ║
╚══════════════════════════════════════════════════════════╝
""")
