import os
import argparse
import shutil
from typing import List


def cleanup_subdirs(parent_dir: str,
                    files_to_delete: List[str] = None,
                    dirs_to_delete: List[str] = None) -> None:
    """
    Clean up specified files and directories in all subdirectories.

    Args:
        parent_dir: Parent directory containing subdirectories to clean
        files_to_delete: List of filenames to delete
        dirs_to_delete: List of directory names to delete
    """
    if files_to_delete is None:
        files_to_delete = ['scraped_data.csv']
    if dirs_to_delete is None:
        dirs_to_delete = ['progress', 'text_files']

    if not os.path.isdir(parent_dir):
        raise ValueError(f"'{parent_dir}' is not a valid directory.")

    # Find immediate subdirectories
    subdirs = [d for d in os.listdir(parent_dir)
               if os.path.isdir(os.path.join(parent_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(parent_dir, subdir)
        print(f"Processing: {subdir_path}")

        # Delete specified files
        for filename in files_to_delete:
            file_path = os.path.join(subdir_path, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"  Deleted file: {filename}")
                except Exception as e:
                    print(f"  Error deleting {filename}: {e}")

        # Delete specified directories
        for dirname in dirs_to_delete:
            dir_path = os.path.join(subdir_path, dirname)
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"  Deleted directory: {dirname}")
                except Exception as e:
                    print(f"  Error deleting directory {dirname}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up specified files and directories in all subdirectories"
    )
    parser.add_argument("parent_dir", help="Parent directory containing subdirectories to clean")
    parser.add_argument("--files", nargs="*", help="List of files to delete")
    parser.add_argument("--dirs", nargs="*", help="List of directories to delete")

    args = parser.parse_args()

    cleanup_subdirs(args.parent_dir, args.files, args.dirs)


if __name__ == "__main__":
    main()
