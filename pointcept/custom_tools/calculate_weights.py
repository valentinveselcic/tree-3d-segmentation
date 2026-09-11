import numpy as np

combined_freq = [0.1199, 0.7952, 0.0849]  
partial_freq = [0.1304, 0.7405, 0.1291]
merged_freq = [0.1152, 0.8198, 0.0650]

freq    = np.array(merged_freq)
weights = 1.0 / freq
weights = weights / weights.mean()
print(f"CE Weights (1/freq): {weights}")

weights = 1.0 / np.sqrt(freq)
weights = weights / weights.mean()
print(f"CE Weights (1/sqrt(freq)): {weights}")

def compute_focal_alpha(class_frequencies: list[float]) -> list[float]:
    assert len(class_frequencies) == 3, "Expected exactly 3 class frequencies"
    assert abs(sum(class_frequencies) - 1.0) < 1e-3, \
        f"Frequencies must sum to 1.0, got {sum(class_frequencies):.4f}"
    assert all(0 < f < 1 for f in class_frequencies), \
        "Each frequency must be in (0, 1)"

    alpha = [round(1.0 - f, 4) for f in class_frequencies]
    return alpha

focal_alpha = compute_focal_alpha(freq)
print(f"Focal Loss (1 - freq): [{', '.join(str(f) for f in focal_alpha)}]")