# U-Net Road Segmentation (ResNet34 backbone)

Binary road segmentation with a U-Net (ResNet34 encoder, via [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) trained with PyTorch Lightning.

## Notebook

[`pytorch-unet-with-resnet34-backbone-segmentation.ipynb`](pytorch-unet-with-resnet34-backbone-segmentation.ipynb) is set up to run on a **free Google Colab GPU runtime** (T4):

- `Runtime -> Change runtime type -> T4 GPU`, then run the cells top to bottom.
- The *Dataset* section has two options for getting the `tgrs-road` dataset onto Colab: downloading it from Kaggle (`kagglehub`), or reading it from a folder already placed in Google Drive. Use whichever fits your setup and skip the other cell.
- Checkpoints are written to Google Drive (`/content/drive/MyDrive/tgrs-road-unet/checkpoints`) so training can resume after a Colab disconnect.

The dataset is expected to have `Train`, `Validation`, and `Test` folders, each containing `image/` and `label/` subfolders.

## Local development

The notebook also runs unmodified on a Kaggle notebook with the dataset attached as `/kaggle/input/tgrs-road`, or locally/on any other Jupyter environment with the dataset on disk and `DATASET_ROOT` pointed at it.
