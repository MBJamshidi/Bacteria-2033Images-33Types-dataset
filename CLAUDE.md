# CLAUDE.md — AI Assistant Guide for Bacteria-2033Images-33Types-dataset

## Repository Purpose

This repository is a **dataset metadata and documentation repository**, not a code project. It describes and provides access to a curated microscopy image dataset of 33 bacterial species (2,033 images total). The actual image data is hosted externally on Google Drive.

The repository exists to:
- Publish dataset metadata and download instructions
- Provide citation information for academic use
- Link to peer-reviewed publications that use or describe the dataset

## Repository Structure

```
Bacteria-2033Images-33Types-dataset/
├── README.md       # Dataset description, download link, citation info
├── LICENSE         # MIT License (copyright Behdad Jamshidi, 2025)
└── CLAUDE.md       # This file
```

There are no source code files, build scripts, tests, or package manifests. All image data (3.4 GB) lives on Google Drive, not in this repository.

## Dataset Facts

| Property | Value |
|---|---|
| Total images | 2,033 |
| Number of classes | 33 bacterial species |
| Image type | High-resolution RGB |
| Sample source | Clinical (blood, urine, skin) |
| Staining method | Gram stain |
| Annotation | Expert laboratory annotation |
| Download size | ~3.4 GB (ZIP) |
| License | MIT |

The dataset has been used in peer-reviewed research on deep transfer learning, federated learning, digital twins, and CNN-based classification for biomedical imaging.

## Key Publications

1. Jamshidi et al. (2023). *Metaverse and microorganism digital twins: A deep transfer learning approach.* Applied Soft Computing, 147, 110798.
2. Jamshidi et al. (2025). *Revolutionizing biological digital twins: Integrating internet of bio-nano things, CNNs, and federated learning.* Computers in Biology and Medicine, 189, 109970.
3. Jamshidi et al. (2024). *CNN-FL for Biotechnology Industry Empowered by Internet-of-BioNano Things and Digital Twins.* IEEE Internet of Things Magazine, 7(5), 54–63.

## Citation Requirement

**Any use of this dataset in research, software, theses, or applications requires a citation.** The canonical BibTeX entry is:

```bibtex
@article{jamshidi2023metaverse,
  title={Metaverse and microorganism digital twins: A deep transfer learning approach},
  author={Jamshidi, Mohammad Behdad and Sargolzaei, Saleh and Foorginezhad, Salimeh and Moztarzadeh, Omid},
  journal={Applied Soft Computing},
  volume={147},
  pages={110798},
  year={2023},
  publisher={Elsevier}
}
```

## Development Workflow

### Branch Convention

- `main` — stable, published state of the repository
- Feature branches follow the pattern `<actor>/<description>` (e.g., `claude/add-claude-documentation-mDfMd`)

### Making Changes

This repository has no build system, tests, or CI pipeline. All changes are documentation-only. Follow these steps:

1. Work on a feature branch, never directly on `main`.
2. Edit `README.md` or other documentation files.
3. Commit with clear, descriptive messages (e.g., `Add dataset statistics table to README`).
4. Push the feature branch: `git push -u origin <branch-name>`.
5. Open a pull request into `main` when the change is ready for review.

### Commit Message Style

The existing commits use simple imperative-mood titles:
- `Initial commit`
- `Update README.md`

Prefer slightly more descriptive messages that state *what* changed and *why*, for example:
- `Add citation BibTeX block to README`
- `Add CLAUDE.md with codebase documentation for AI assistants`

### What NOT to Do

- Do not add code, notebooks, or scripts to this repository unless they are directly needed for dataset tooling agreed upon by the maintainer.
- Do not commit image data or large binary files; the dataset is distributed via Google Drive.
- Do not push directly to `main`.
- Do not modify the LICENSE file without explicit maintainer approval.

## Working with the Dataset (for ML/AI Tasks)

When helping users build models on this dataset, keep in mind:

- **Download**: Users must download the ZIP from Google Drive (link in README.md) and extract locally. There is no programmatic API.
- **Expected directory layout after extraction**: Typically one folder per class label (33 folders), each containing JPEG/PNG images. Confirm with the actual ZIP contents.
- **Recommended frameworks**: PyTorch (`torchvision.datasets.ImageFolder`) or TensorFlow/Keras (`image_dataset_from_directory`) both work naturally with per-class folder layouts.
- **Class imbalance**: 2,033 images across 33 classes averages ~62 images per class. Expect imbalance; use weighted loss or augmentation accordingly.
- **Preprocessing**: Images are Gram-stained RGB microscopy images. Standard ImageNet normalization is a reasonable starting point for transfer learning.
- **Train/val/test splits**: No official split is defined in the repository. A common practice in the published papers is stratified splitting to preserve class proportions.

## Maintainer

Mohammad Behdad Jamshidi ([@MBJamshidi](https://github.com/MBJamshidi))
