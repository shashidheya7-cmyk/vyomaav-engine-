with open("vyomaa/perception/sam2_worker.py", "r") as f:
    code = f.read()

# Replace torch.cuda.reset_peak_memory_stats(self.device) with a safe device index integer or generic call
old_line = "torch.cuda.reset_peak_memory_stats(self.device)"
new_line = "torch.cuda.reset_peak_memory_stats(self.device.index if self.device.index is not None else 0)"

if old_line in code:
    code = code.replace(old_line, new_line)

with open("vyomaa/perception/sam2_worker.py", "w") as f:
    f.write(code)

print("[✓] Patched torch.cuda.reset_peak_memory_stats device argument in sam2_worker.py")
