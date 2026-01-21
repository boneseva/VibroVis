#!/usr/bin/env python3

"""
Converts a directory of .wav files to .mp3 files, preserving the
subdirectory structure.

It replaces the top-level input directory name (e.g., 'audio_16k')
with a new name (e.g., 'mp3') in the output path.

Example:
  Input:  /path/to/audio_16k/vocals/track1.wav
  Output: /path/to/mp3/vocals/track1.mp3

Usage:
  python wav_to_mp3_converter.py /path/to/your/audio_16k
"""

import sys
import argparse
from pathlib import Path
from pydub import AudioSegment


def convert_wav_to_mp3(input_dir_path: Path, output_dir_name: str = "mp3"):
    """
    Finds all .wav files in input_dir_path, converts them to .mp3,
    and saves them in a new directory structure.

    Args:
        input_dir_path (Path): The root directory to search for .wav files
                               (e.g., /path/to/audio_16k).
        output_dir_name (str): The name for the new top-level output
                               directory (e.g., 'mp3').
    """

    # --- 1. Basic Input Validation ---
    if not input_dir_path.exists():
        print(f"Error: Input directory not found: {input_dir_path}")
        sys.exit(1)
    if not input_dir_path.is_dir():
        print(f"Error: Input path is not a directory: {input_dir_path}")
        sys.exit(1)

    # --- 2. Define Output Root Directory ---
    # Create the new output directory (e.g., 'mp3')
    # alongside the input directory (e.g., 'audio_16k')
    # /path/to/audio_16k -> /path/to/mp3
    output_root_dir = input_dir_path.parent / output_dir_name

    print(f"Input directory:  {input_dir_path}")
    print(f"Output directory: {output_root_dir}")
    print("-" * 30)

    # --- 3. Find and Process All .wav Files ---
    wav_files = list(input_dir_path.rglob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {input_dir_path}")
        return

    print(f"Found {len(wav_files)} .wav files. Starting conversion...")

    for wav_file in wav_files:
        try:
            # --- 4. Determine Output Path ---

            # Get the path relative to the input_dir
            # e.g., 'vocals/track1.wav'
            relative_path = wav_file.relative_to(input_dir_path)

            # Create the new target path
            # e.g., /path/to/mp3/vocals/track1.wav
            target_mp3_path_with_wav_ext = output_root_dir / relative_path

            # Change the extension to .mp3
            # e.g., /path/to/mp3/vocals/track1.mp3
            target_mp3_path = target_mp3_path_with_wav_ext.with_suffix(".mp3")

            # --- 5. Create Output Subdirectory (if needed) ---
            target_mp3_path.parent.mkdir(parents=True, exist_ok=True)

            # --- 6. Perform Conversion ---
            print(f"Converting: {wav_file.name}...")

            # Load the WAV file
            audio = AudioSegment.from_wav(str(wav_file))

            # Export as MP3
            # You can add parameters here, e.g., bitrate="192k"
            audio.export(str(target_mp3_path), format="mp3")

            print(f"  -> Saved: {target_mp3_path}")

        except Exception as e:
            print(f"!! FAILED to convert {wav_file.name}: {e}")

    print("-" * 30)
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of .wav files to .mp3, preserving file structure.",
        epilog="Example: python wav_to_mp3_converter.py ./audio_16k"
    )

    # Add the positional argument for the input directory
    parser.add_argument(
        "--input_directory",
        type=str,
        help="The path to the top-level directory containing .wav files (e.g., 'audio_16k')."
    )

    args = parser.parse_args()

    # Convert the string path to a Path object and resolve it
    # This makes it an absolute path, removing ambiguity.
    try:
        input_dir = Path(args.input_directory).resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: Input directory not found: {args.input_directory}")
        sys.exit(1)

    # Run the conversion
    convert_wav_to_mp3(input_dir, output_dir_name="mp3")


if __name__ == "__main__":
    main()