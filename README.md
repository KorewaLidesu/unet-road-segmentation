# U-Net Road Segmentation (ResNet34 backbone)

Binary road segmentation with a U-Net (ResNet34 encoder, via [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) trained with PyTorch Lightning on the [DeepGlobe Road Extraction dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset).

## Notebook

[`pytorch-unet-with-resnet34-backbone-segmentation.ipynb`](pytorch-unet-with-resnet34-backbone-segmentation.ipynb) is set up to run on a **free Google Colab GPU runtime** (T4):

- `Runtime -> Change runtime type -> T4 GPU`, then run the cells top to bottom.
- The *Dataset* section downloads the DeepGlobe dataset via `kagglehub` and splits its labeled `train/` pairs into a `Train`/`Validation`/`Test` layout (the raw `valid/`/`test/` folders are unlabeled Kaggle competition holdouts, so they aren't usable directly). The prepared split and checkpoints are both cached on Google Drive so a fresh Colab session reuses them instead of re-downloading/reprocessing, and training resumes automatically if the session gets cut off.

## Local development

The notebook also runs on a Kaggle notebook with the dataset attached as `/kaggle/input/deepglobe-road-extraction-dataset`, or locally with `RAW_DATASET_ROOT`/`PREPARED_DATASET_ROOT` pointed at whatever paths you want.
