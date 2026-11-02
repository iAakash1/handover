# Archive: original files and previous versions

Two kinds of historical material, kept so nothing is lost. Nothing in `projects/` depends on this folder.

- [`previous_versions/`](previous_versions/): earlier versions of final projects, kept as they were: CottonCrop of
  4 and 6 Jan and the 27 Jan integrated pipeline ([01](previous_versions/01_cnn_keras_tomato/README.md)), an earlier
  BLIP-2 quick-train script ([04](previous_versions/04_llava_blip2/blip2_rerun_june_earlier_copies_2026-06-12/README.md)),
  the 13 Apr crop app and the 9 Apr Qwen script variant ([06](previous_versions/06_qwen25vl_plantvillage/README.md)),
  and the 15 May PlantDx ([07](previous_versions/07_plantdx_plantvillage/README.md)).
- `original_files/`: the originals of files edited for the handover (below).

## Original files

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

The text of these copies is unedited. Their relative links pointed into the repository each file came from, so they
now point to the same files on GitHub (github.com/iAakash1/TIH-IoT, or github.com/iAakash1/experimentation for the
workstation copy of the TIH project README) and open from here.
