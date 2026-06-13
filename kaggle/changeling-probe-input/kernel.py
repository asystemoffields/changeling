"""Diagnostic: where does Kaggle mount kernel_sources outputs for script
kernels? Prints the /kaggle/input tree (3 levels)."""
import os

root = "/kaggle/input"
if not os.path.isdir(root):
    print("NO /kaggle/input")
else:
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        print(("  " * depth) + cur + "/")
        if depth >= 3:
            dirs[:] = []
        for f in files[:20]:
            print(("  " * (depth + 1)) + f)
