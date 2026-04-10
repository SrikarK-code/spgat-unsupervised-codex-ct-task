import os
from pathlib import Path
from rich.console import Console
from rich.tree import Tree

def add_to_tree(dir_path: Path, tree: Tree):
    # Sort paths: directories first, then files, alphabetically
    try:
        paths = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return

    for path in paths:
        # Skip hidden files, python caches, and STHD heavy patch folders
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        
        # Explicitly hide the 'patches' folder contents to prevent console flooding
        if path.is_dir() and path.name == "patches":
            tree.add("[dim italic]📁 patches/ (contents hidden)[/dim italic]")
            continue
            
        if path.is_dir():
            # Add directory branch and recurse
            branch = tree.add(f"[bold blue]📁 {path.name}/[/bold blue]")
            add_to_tree(path, branch)
        else:
            # Color-code files based on extension for rapid visual parsing
            if path.suffix == ".png":
                style = "bold magenta"
                icon = "🖼️"
            elif path.suffix in [".txt", ".log"]:
                style = "yellow"
                icon = "📄"
            elif path.suffix in [".tsv", ".csv"]:
                style = "cyan"
                icon = "📊"
            elif path.suffix == ".py":
                style = "bold green"
                icon = "🐍"
            elif path.suffix == ".sh":
                style = "bold green"
                icon = "��"
            else:
                style = "white"
                icon = "📄"
                
            tree.add(f"[{style}]{icon} {path.name}[/{style}]")

if __name__ == "__main__":
    console = Console()
    
    # Target directory provided
    target_dir = Path("/hpc/home/vk93/lab_vk93/sthd-codex/")
    
    if not target_dir.exists():
        console.print(f"[bold red]Error: Directory {target_dir} does not exist.[/bold red]")
    else:
        root_tree = Tree(f"[bold red]📂 {target_dir.name}/[/bold red]")
        add_to_tree(target_dir, root_tree)
        console.print(root_tree)
