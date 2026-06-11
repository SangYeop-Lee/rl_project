import platform
import subprocess
import torch


def sysctl(name):
    try:
        return (
            subprocess.check_output(["sysctl", "-n", name])
            .decode()
            .strip()
        )
    except Exception:
        return "N/A"


print("=" * 80)
print("System")
print("=" * 80)
print("macOS           :", platform.mac_ver()[0])
print("Machine         :", platform.machine())
print()

print("=" * 80)
print("CPU")
print("=" * 80)
print("CPU Brand       :", sysctl("machdep.cpu.brand_string"))
print("CPU Cores       :", sysctl("hw.ncpu"))
print("Perf Cores      :", sysctl("hw.perflevel0.physicalcpu"))
print("Eff Cores       :", sysctl("hw.perflevel1.physicalcpu"))
print()

print("=" * 80)
print("Apple Silicon")
print("=" * 80)
print("Chip            :", sysctl("machdep.cpu.brand_string"))
print("Memory          :", f"{int(sysctl('hw.memsize')) / 1024**3:.1f} GB")
print()

print("=" * 80)
print("PyTorch")
print("=" * 80)
print("Torch Version   :", torch.__version__)
print("MPS Available   :", torch.backends.mps.is_available())
print("MPS Built       :", torch.backends.mps.is_built())
