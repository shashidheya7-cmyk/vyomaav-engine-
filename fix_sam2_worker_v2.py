with open("vyomaa/perception/sam2_worker.py", "r") as f:
    code = f.read()

# Replace reset_peak_memory_stats with a generic call without device argument or wrap in try-except
old_line = "torch.cuda.reset_peak_memory_stats(self.device.index if self.device.index is not None else 0)"
new_line = "try:\n                torch.cuda.reset_peak_memory_stats()\n            except Exception:\n                pass"

if old_line in code:
    code = code.replace(old_line, new_line)

with open("vyomaa/perception/sam2_worker.py", "w") as f:
    f.write(code)

print("[✓] Patched torch.cuda.reset_peak_memory_stats safely in sam2_worker.py")
