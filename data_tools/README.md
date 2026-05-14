# data_tools/

Utility scripts for preparing dataset splits and organizing image files.

## Scripts

| Script | Original name | When to use |
|---|---|---|
| `generate_csv_from_folder.py` | `GenerateCSVForDatsetFolder.py` (typo fixed) | Generate a CSV split file from a directory of images organized by class subfolder |
| `separate_images_by_class.py` | `SeparateImageToFolderWithCSV.py` | Sort flat image directory into per-class subfolders using an existing CSV label file |

## Typical workflow

1. Download raw dataset images into a flat directory
2. Run `generate_csv_from_folder.py` to produce train/test CSV splits
3. (Optional) Run `separate_images_by_class.py` to reorganize images into class subfolders
4. Move generated CSVs to the appropriate `data/splits/<dataset>/` directory
