# Original files, exactly as they were

To make the projects run on any machine, a few files were edited for the handover: dataset paths can now be set
with environment variables, image paths in the PlantVLM data files became relative, header comments were added, and
the month READMEs got working links. This folder keeps the **unedited originals** of every such file, so nothing
is lost:

- `original_files/workstation/...`: as found on the TIH workstation (`C:\Users\Admin\Desktop\...`): VLM scripts,
  SoyNet configs, the PlantVLM data files with their original absolute paths, the TIH project's app runtime.
- `original_files/TIH-IoT/...`: as published in github.com/iAakash1/TIH-IoT (README pages, the October evaluation
  script, the same originals as above where TIH-IoT was the source). `.gitignore` / `.gitattributes` files are
  stored as `dot_gitignore.txt` / `dot_gitattributes.txt` so they don't change how git treats this repository.

To see what changed in a file, compare it with its counterpart in `projects/` (same file name).
