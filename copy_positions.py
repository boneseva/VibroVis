import os
import shutil

def copy_positions_folder(original_root, new_root):
    for dirpath, dirnames, filenames in os.walk(original_root):
        if "positions" in dirnames:
            src_positions_path = os.path.join(dirpath, "positions")
            # The parent path relative to original root
            rel_parent_dir = os.path.relpath(os.path.dirname(src_positions_path), original_root)
            dest_parent_dir = os.path.join(new_root, rel_parent_dir)
            dest_positions_path = os.path.join(dest_parent_dir, "positions")
            os.makedirs(dest_parent_dir, exist_ok=True)
            if os.path.exists(dest_positions_path):
                shutil.rmtree(dest_positions_path)  # Remove if exists to avoid conflict
            shutil.copytree(src_positions_path, dest_positions_path)
            print(f"Copied: {src_positions_path} to {dest_positions_path}")


copy_positions_folder(
     "/mnt/voz/Vibroscape/Nonannotated recordings/Rok spring summer/16k",
     "/mnt/voz/Vibroscape/Nonannotated recordings/Rok spring summer/16k/mp3")
