import torch, sys

paths = [
    r"checkpoints\best_model.pt",
    r"checkpoints\best_esp32_11class.pt",
    r"outputs\best_model.pt",
]

for p in paths:
    print(f"\n=== {p} ===")
    try:
        s = torch.load(p, map_location="cpu", weights_only=False)
        print("keys:", list(s.keys()))
        for k in ["num_classes", "labels", "activity_labels", "normalization", "normalization_stats"]:
            if k in s:
                v = s[k]
                if isinstance(v, dict) and any(x in v for x in ["mean", "means", "std", "stds"]):
                    print(f"{k}: present dict")
                elif isinstance(v, list):
                    print(f"{k}: {v}")
                else:
                    print(f"{k}: {v}")
    except Exception as e:
        print("error:", e)
