# Bacteria-2033Images-33Types-dataset

Microscopy image dataset of 33 bacteria species for machine learning and deep learning research.

## 🧫 33-Class Bacterial Microscopy Image Dataset (2033 Images)

This dataset contains 2,033 high-resolution RGB images of **33 bacterial species**, collected from clinical samples (blood, urine, skin), Gram-stained, and annotated by laboratory experts.

![bacteria](https://github.com/user-attachments/assets/7a3044b8-8f1d-427b-8e32-169414c42652)

---

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

---

## 📦 Download the Dataset

🔗 [Click here to download the dataset (3.4 GB, Google Drive)](https://drive.google.com/file/d/1aR7Dz11wKV3t7awnnnO32UE_37MYF6wX/view?usp=sharing)

---

## 📚 Key Publications

1. Jamshidi, M. B., Sargolzaei, S., Foorginezhad, S., & Moztarzadeh, O. (2023). Metaverse and microorganism digital twins: A deep transfer learning approach. *Applied Soft Computing*, 147, 110798.
   https://www.sciencedirect.com/science/article/pii/S1568494623008165

2. Jamshidi, M.B., Hoang, D.T., Nguyen, D.N., Niyato, D. and Warkiani, M.E. (2025). Revolutionizing biological digital twins: Integrating internet of bio-nano things, convolutional neural networks, and federated learning. *Computers in Biology and Medicine*, 189, 109970.
   https://www.sciencedirect.com/science/article/pii/S001048252500321X

3. Jamshidi, M., Hoang, D.T. and Nguyen, D.N. (2024). CNN-FL for Biotechnology Industry Empowered by Internet-of-BioNano Things and Digital Twins. *IEEE Internet of Things Magazine*, 7(5), 54–63.
   https://ieeexplore.ieee.org/abstract/document/10643983

---

## 📥 Citation

If you use this dataset in any research paper, project, book, thesis, or any other scholarly or commercial work, please cite it as follows:

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

---

## 🤖 Working with the Dataset (ML/AI)

- **Download**: Download the ZIP from Google Drive (link above) and extract locally. There is no programmatic API.
- **Directory layout**: One folder per class label (33 folders), each containing the class images.
- **Recommended frameworks**: PyTorch (`torchvision.datasets.ImageFolder`) or TensorFlow/Keras (`image_dataset_from_directory`) both work naturally with this per-class folder layout.
- **Class imbalance**: 2,033 images across 33 classes averages ~62 images per class. Expect imbalance; use weighted loss or data augmentation accordingly.
- **Preprocessing**: Standard ImageNet normalization is a reasonable starting point for transfer learning on these Gram-stained RGB images.
- **Train/val/test splits**: No official split is defined. Stratified splitting to preserve class proportions is recommended, consistent with published papers.

---

## 🗂️ Repository Structure

```
Bacteria-2033Images-33Types-dataset/
├── README.md       # Dataset description, download link, citation info
└── LICENSE         # MIT License
```

The image data (3.4 GB) is distributed via Google Drive, not stored in this repository.

---

## 🛠️ Development Workflow (for Contributors)

- Work on a feature branch, never directly on `main`.
- All changes in this repository are documentation-only (no code or binary data).
- Do not commit image data or large binary files.
- Push the feature branch and open a pull request into `main` when ready.

---

## Maintainer

Mohammad Behdad Jamshidi ([@MBJamshidi](https://github.com/MBJamshidi))
