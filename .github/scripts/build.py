#!/usr/bin/env python3
"""
Build script for marimo notebooks.

This script exports marimo notebooks to static HTML format and generates
an index.html file that lists all the notebooks.
"""

import os
import subprocess
import argparse
from typing import List
from pathlib import Path


def export_html_static(notebook_path: str, output_dir: str) -> bool:
    """Export a single marimo notebook to static HTML format.

    This function takes a marimo notebook (.py file) and exports it to a static HTML file,
    which includes the code and pre-rendered outputs.

    Args:
        notebook_path (str): Path to the marimo notebook (.py file) to export
        output_dir (str): Directory where the exported HTML file will be saved

    Returns:
        bool: True if export succeeded, False otherwise
    """
    # Convert .py extension to .html for the output file
    output_path: str = notebook_path.replace(".py", ".html")

    # Base command for static marimo export
    # The command is changed from 'html-wasm' to 'html'
    cmd: List[str] = ["marimo", "export", "html"]

    print(f"Exporting {notebook_path} to {output_path} as static HTML")

    try:
        # Create full output path and ensure directory exists
        output_file: str = os.path.join(output_dir, output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Add notebook path and output file to command
        cmd.extend([notebook_path, "-o", output_file])

        # Run marimo export command
        print(cmd)
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        # Handle marimo export errors
        print(f"Error exporting {notebook_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def generate_index(all_notebooks: List[str], output_dir: str) -> None:
    """Generate an index.html file that lists all the notebooks."""
    # This function does not need any changes.
    print("Generating index.html")
    index_path: str = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(index_path, "w") as f:
            f.write(
                """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>marimo</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  </head>
  <body class="font-sans max-w-2xl mx-auto p-8 leading-relaxed">
    <div class="mb-8">
      <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo" class="h-20" />
    </div>
    <div class="grid gap-4">
"""
            )
            for notebook in all_notebooks:
                notebook_name: str = notebook.split("/")[-1].replace(".py", "")
                display_name: str = notebook_name.replace("_", " ").title()
                f.write(
                    f'      <div class="p-4 border border-gray-200 rounded">\n'
                    f'        <h3 class="text-lg font-semibold mb-2">{display_name}</h3>\n'
                    f'        <div class="flex gap-2">\n'
                    f'          <a href="{notebook.replace(".py", ".html")}" class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">Open Notebook</a>\n'
                    f"        </div>\n"
                    f"      </div>\n"
                )
            f.write(
                """    </div>
  </body>
</html>"""
            )
    except IOError as e:
        print(f"Error generating index.html: {e}")


def main() -> None:
    """Main function to build marimo notebooks."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    args: argparse.Namespace = parser.parse_args()

    all_notebooks: List[str] = []
    # Look for notebooks in both 'notebooks' and 'apps' directories
    for directory in ["notebooks", "apps"]:
        dir_path: Path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        all_notebooks.extend(str(path) for path in dir_path.rglob("*.py"))

    if not all_notebooks:
        print("No notebooks found!")
        return

    # Export each notebook to static HTML format.
    # The distinction between 'app' and 'notebook' mode is no longer needed.
    for nb in all_notebooks:
        export_html_static(nb, args.output_dir)

    generate_index(all_notebooks, args.output_dir)


if __name__ == "__main__":
    main()
