from rich.tree import Tree
from rich import print
import os

def walk(dir, tree):
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        branch = tree.add(name)
        if os.path.isdir(path):
            walk(path, branch)

t = Tree("sthd-codex")
walk("/hpc/home/vk93/lab_vk93/sthd-codex/", t)
print(t)
