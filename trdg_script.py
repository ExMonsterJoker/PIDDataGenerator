import yaml
import os
import argparse
import random
import string
import time
import re
import pandas as pd
import numpy as np
from collections import Counter

# It's good practice to handle potential import errors for optional features
try:
    from trdg.generators import (
        GeneratorFromStrings,
        GeneratorFromRandom,
    )
except ImportError:
    print("Error: 'trdg' library not found. Please install it using 'pip install trdg'")
    exit()


class TRDGOrchestrator:
    """
    A class to orchestrate the generation of synthetic text data using TRDG.
    It handles configuration, generator selection, execution, and summary reporting.
    """

    def __init__(self, config_path):
        """
        Initializes the orchestrator.
        :param config_path: Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.stats = {'count': 0, 'char_counts': Counter()}

        # Ensure output directory exists
        self.output_dir = self.config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Acknowledge the known warning from trdg, which is normal.
        print(
            "Note: If you see 'Missing modules for handwritten text generation', it's a standard TRDG startup message and can be ignored unless you need handwritten text.")

    def _load_config(self, config_path):
        """Loads the YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at '{config_path}'")
            exit()

    def _setup_data_splits(self):
        """Setup data splitting configuration and create directories."""
        self.enable_split = self.config.get('enable_data_split', False)

        if not self.enable_split:
            return

        self.split_ratios = self.config.get('split_ratios', {
            'train': 0.7,
            'valid': 0.2,
            'test': 0.1
        })

        # Normalize ratios to sum to 1
        total_ratio = sum(self.split_ratios.values())
        self.split_ratios = {k: v / total_ratio for k, v in self.split_ratios.items()}

        # Create split directories
        self.split_dirs = {}
        for split_name in self.split_ratios.keys():
            if self.split_ratios[split_name] > 0:
                split_dir = os.path.join(self.output_dir, f"{split_name}_data")
                os.makedirs(split_dir, exist_ok=True)
                self.split_dirs[split_name] = split_dir

        print(f"Data splits configured: {self.split_ratios}")

    def _get_split_assignment(self, index, total_count):
        """Determine which split an image belongs to based on index."""
        if not self.enable_split:
            return None, self.output_dir

        # Create cumulative probabilities
        cumulative = 0
        rand_val = index / total_count  # Deterministic based on index

        for split_name, ratio in self.split_ratios.items():
            if ratio > 0:
                cumulative += ratio
                if rand_val < cumulative:
                    return split_name, self.split_dirs[split_name]

        # Fallback to last split
        last_split = list(self.split_ratios.keys())[-1]
        return last_split, self.split_dirs[last_split]

    def _generate_string_from_template(self, template, custom_symbols, text_case):
        """Generates a random string based on a template format."""
        result = []
        # UPDATED: Added FRAC to the regex
        parts = re.split(r'(\d+x(?:NUM|CHAR|SYM|FRAC))', template)

        char_source = string.ascii_letters
        if text_case == 'upper':
            char_source = string.ascii_uppercase
        elif text_case == 'lower':
            char_source = string.ascii_lowercase

        # NEW: Defined a list of common Unicode fraction characters
        fractions = ['½', '⅓', '⅔', '¼', '¾', '⅕', '⅖', '⅗', '⅘', '⅙', '⅚', '⅛', '⅜', '⅝', '⅞']

        # UPDATED: Added FRAC to the char_map
        char_map = {
            'NUM': string.digits,
            'CHAR': char_source,
            'SYM': custom_symbols if custom_symbols else '_-.,',
            'FRAC': fractions
        }

        for part in parts:
            if not part:
                continue

            # UPDATED: Added FRAC to the regex
            match = re.match(r'(\d+)x(NUM|CHAR|SYM|FRAC)', part)
            if match:
                count = int(match.group(1))
                type_key = match.group(2)
                result.append(''.join(random.choices(char_map[type_key], k=count)))
            else:
                result.append(part)

        return "".join(result)

    def _apply_text_case(self, text_list):
        """Applies the configured text case to a list of strings."""
        case_setting = self.config.get('text_case', 'mixed').lower()
        if case_setting == 'upper':
            return [s.upper() for s in text_list]
        elif case_setting == 'lower':
            return [s.lower() for s in text_list]
        return text_list  # Default is 'mixed'

    def _prepare_text_source(self):
        """Prepares the list of strings to be generated based on the config."""
        count = self.config.get('count', 100)
        template = self.config.get('text_format_template')
        source_file = self.config.get('source_file_path')

        template_word_ratio = self.config.get('template_word_ratio', 0.0)
        text_case = self.config.get('text_case', 'mixed').lower()
        custom_symbols = self.config.get('custom_char_set', '')

        strings = []

        # --- Logic for generating the source text list ---

        # Case 1: Mix of template and file words
        if template and source_file and 0 < template_word_ratio < 1:
            print(f"Info: Mixing template and file words with a {template_word_ratio * 100}% template ratio.")
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_words = [line.strip() for line in f.readlines() if line.strip()]
                if not source_words:
                    print("Warning: Source file is empty. Cannot mix.")
                else:
                    for _ in range(count):
                        if random.random() < template_word_ratio:
                            strings.append(self._generate_string_from_template(template, custom_symbols, text_case))
                        else:
                            strings.append(random.choice(source_words))
            except FileNotFoundError:
                print(f"Error: Source file '{source_file}' not found. Cannot mix.")

        # Case 2: Only template words
        elif template:
            print("Info: Using text template engine.")
            strings = [self._generate_string_from_template(template, custom_symbols, text_case) for _ in range(count)]
            return self._apply_text_case(strings)

        # Case 3: Only file-based words (including priority logic)
        elif source_file:
            print("Info: Using text from file(s).")
            priority_file = self.config.get('priority_word_file')
            priority_ratio = self.config.get('priority_ratio', 0.0)
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_words = [line.strip() for line in f.readlines() if line.strip()]

                if priority_file and priority_ratio > 0:
                    with open(priority_file, 'r', encoding='utf-8') as f:
                        priority_words = [line.strip() for line in f.readlines() if line.strip()]
                    for _ in range(count):
                        chosen_list = priority_words if random.random() < priority_ratio and priority_words else source_words
                        if chosen_list:
                            strings.append(random.choice(chosen_list))
                else:
                    if source_words:
                        strings = [random.choice(source_words) for _ in range(count)]

            except FileNotFoundError as e:
                print(f"Error: Word file not found - {e}")

        # Case 4: Fallback to fully random generation if no other source is defined
        else:
            print("Info: No text source specified. Falling back to random text generation.")
            return None

        if not strings:
            print("Warning: Text source generation resulted in an empty list.")
            return []

        return self._apply_text_case(strings)

    def _get_common_args(self):
        """Returns a dictionary of arguments common to ALL generators."""
        common_args = {
            'language': self.config.get('language', 'en'),
            'skewing_angle': self.config.get('skewing_angle', 0),
            'blur': self.config.get('blur', 0),
            'background_type': self.config.get('background_type', 1),
            'distorsion_type': self.config.get('distorsion_type', 0),
            'text_color': self.config.get('text_color', '') or '#282828',
            'margins': tuple(self.config.get('margins', {}).values()),
            'fit': self.config.get('fit', False)
        }
        return common_args

    def _create_generator(self, strings=None):
        """
        Creates a TRDG generator instance, carefully assigning arguments
        to the correct generator type to avoid TypeErrors.
        """
        args = self._get_common_args()

        args['width'] = self.config.get('image_size', {}).get('width', -1)
        height_setting = self.config.get('image_size', {}).get('height', 32)
        if isinstance(height_setting, list) and self.config.get('random_height'):
            args['size'] = random.randint(height_setting[0], height_setting[1])
        else:
            args['size'] = height_setting if isinstance(height_setting, int) else height_setting[0]

        if self.config.get('font_file'):
            font_path = os.path.join(self.config.get('font_dir', 'fonts'), self.config['font_file'])
            args['fonts'] = [font_path] if os.path.exists(font_path) else []
            if not args['fonts']:
                print(f"Warning: Specified font_file '{self.config['font_file']}' not found. Using random fonts.")
        else:
            args['fonts'] = []

        if strings is not None:
            allowed_keys = [
                'strings', 'fonts', 'language', 'size', 'skewing_angle', 'random_skew',
                'blur', 'random_blur', 'background_type', 'distorsion_type', 'width',
                'text_color', 'margins', 'fit'
            ]
            args['strings'] = strings
            args['random_skew'] = self.config.get('random_skew', False)
            args['random_blur'] = self.config.get('random_blur', False)
            gfs_args = {key: value for key, value in args.items() if key in allowed_keys}
            return GeneratorFromStrings(**gfs_args)
        else:
            args['count'] = self.config.get('count', 100)
            args['length'] = self.config.get('length', 1)
            args['allow_space'] = self.config.get('allow_space', True)
            args['random_skew'] = self.config.get('random_skew', False)
            args['random_blur'] = self.config.get('random_blur', False)
            args['character_spacing'] = self.config.get('character_spacing', 0)
            if self.config.get('custom_char_set'):
                args['random_sequences'] = True
                args['source'] = self.config.get('custom_char_set')
            text_case = self.config.get('text_case', 'mixed').lower()
            if text_case == 'upper' or text_case == 'lower':
                args['use_upper_case'] = text_case == 'upper'
            return GeneratorFromRandom(**args)

    def _generate_summary(self, duration):
        """Generates a summary report of the generation process."""
        summary = ["=" * 30, "      GENERATION SUMMARY", "=" * 30]
        summary.append(f"Total Images Generated: {self.stats['count']}")
        summary.append(f"Total Generation Time: {duration:.2f} seconds")
        if self.stats['count'] > 0:
            summary.append(f"Average Time per Image: {duration / self.stats['count']:.4f} seconds")
        summary.append("\n--- Configuration Snapshot ---")
        for key, value in self.config.items():
            summary.append(f"{key}: {value}")
        summary.append("\n--- Character Frequency ---")
        for char, count in self.stats['char_counts'].most_common():
            display_char = "' '" if char == " " else char
            summary.append(f"  {display_char:<5}: {count}")
        return "\n".join(summary)

    def run(self):
        """Updated run method with data splitting and CSV output."""
        start_time = time.time()

        # Add this line after your existing initialization
        self._setup_data_splits()

        strings_to_generate = self._prepare_text_source()
        generation_count = len(strings_to_generate) if strings_to_generate is not None else self.config.get('count',
                                                                                                            100)
        self.config['count'] = generation_count

        if generation_count == 0:
            print("Error: No text to generate. Please check your configuration and source files.")
            return

        print(f"Starting data generation for {generation_count} images.")

        # Get image format from config
        image_format = self.config.get('image_format', 'jpg').lower()

        # Initialize data collection for CSV
        csv_data = []

        # If splitting is enabled, create separate CSV data for each split
        if self.enable_split:
            split_csv_data = {split_name: [] for split_name in self.split_dirs.keys()}

        for i in range(generation_count):
            try:
                current_string_list = [strings_to_generate[i]] if strings_to_generate else None
                current_generator = self._create_generator(current_string_list)

                if current_generator is None:
                    continue

                gen_output = next(current_generator, None)
                if gen_output is None:
                    continue

                img, lbl = None, None
                if isinstance(gen_output, tuple):
                    img, lbl = gen_output[0], gen_output[1]
                else:
                    img = gen_output
                    lbl = current_string_list[0] if current_string_list else ""

                if not img or not lbl:
                    continue

                self.stats['count'] += 1
                self.stats['char_counts'].update(lbl)

                # Determine split assignment and save location
                split_name, save_dir = self._get_split_assignment(i, generation_count)

                # Create filename with proper extension
                image_name = f"image_{i}.{image_format}"
                image_path = os.path.join(save_dir, image_name)

                # Save image
                img.save(image_path)

                # Add to CSV data
                if self.enable_split and split_name:
                    split_csv_data[split_name].append({
                        'filename': image_name,
                        'words': lbl
                    })
                else:
                    csv_data.append({
                        'filename': image_name,
                        'words': lbl
                    })

                print(f"\rGenerated {i + 1}/{generation_count}", end="", flush=True)

            except Exception as e:
                text_context = strings_to_generate[i] if strings_to_generate and i < len(
                    strings_to_generate) else 'random'
                print(f"\nError during generation of item {i} ('{text_context}'): {e}")
                print("Skipping this item and continuing...")

        # Save CSV files
        if self.enable_split:
            for split_name, data in split_csv_data.items():
                if data:  # Only create CSV if there's data
                    df = pd.DataFrame(data)
                    csv_path = os.path.join(self.split_dirs[split_name], f"{split_name}_labels.csv")
                    df.to_csv(csv_path, sep='\t', index=False)
                    print(f"\n{split_name} labels saved to: {csv_path}")
        else:
            # Create single CSV file
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = os.path.join(self.output_dir, 'labels.csv')
                df.to_csv(csv_path, sep='\t', index=False)
                print(f"\nLabels saved to: {csv_path}")

        duration = time.time() - start_time
        print(f"\n\nGeneration complete in {duration:.2f} seconds.")

        summary_text = self._generate_summary(duration)
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Summary report saved to '{summary_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic text data for TRDG.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the YAML config file.')
    cli_args = parser.parse_args()

    orchestrator = TRDGOrchestrator(config_path=cli_args.config)
    orchestrator.run()
