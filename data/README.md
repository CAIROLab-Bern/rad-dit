# Data

This folder holds symlinks to your local data. Git ignores everything here except this file.
The commands expect:

- `radimagenet/CT/` and `radimagenet/MR/`, with `<modality>/<bodypart>/<pathology>/*.png`
- `brats24/BraTS-GLI/training_data1_v2`, `training_data_additional` and `validation_data`
- `imagenet/train/<synset>/` (only for mixed training)

The datasets are not part of this repository. Get them from their providers under their own terms.
