import numpy as np

# ── Assembly 1 ──────────────────────────────────────────────────────────────
times_1 = np.array([108, 244, 244, 94, 242, 100, 110, 166, 184, 235, 94, 166])
t_mean_1 = times_1.mean()
t_std_1 = times_1.std()
scores_1 = np.abs(times_1 - t_mean_1) / t_std_1

print("=== Assembly 1 ===")
print(f"Mean time:  {t_mean_1:.2f} mm")
print(f"Std time:   {t_std_1:.2f} mm")
print(f"Total time: {times_1.sum()} mm\n")

joints_1 = [f"j{i+1}" for i in range(len(times_1))]

for n in [3, 4, 5]:
    opt = times_1.sum() / n
    ps = times_1 / opt
    print(f"\n  Phase score (P={n}, opt_phase={opt:.2f} mm):")
    for j, p in zip(joints_1, ps):
        print(f"    {j}: {p:.4f}")

# ── Assembly 2 ──────────────────────────────────────────────────────────────
times_2 = np.array(
    [200, 75, 93, 560, 75, 560, 133, 240, 240, 158, 101, 52, 52, 50, 50, 25, 25]
)
t_mean_2 = times_2.mean()
t_std_2 = times_2.std()
scores_2 = np.abs(times_2 - t_mean_2) / t_std_2

print("\n\n=== Assembly 2 ===")
print(f"Mean time:  {t_mean_2:.2f} mm")
print(f"Std time:   {t_std_2:.2f} mm")
print(f"Total time: {times_2.sum()} mm\n")

joints_2 = [f"j{i+1}" for i in range(len(times_2))]

for n in [3, 4, 5]:
    opt = times_2.sum() / n
    ps = times_2 / opt
    print(f"\n  Phase score (P={n}, opt_phase={opt:.2f} mm):")
    for j, p in zip(joints_2, ps):
        print(f"    {j}: {p:.4f}")
