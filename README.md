# InvVis

This is the model code for [InvVis: Large-Scale Data Embedding for Invertible Visualization](https://arxiv.org/pdf/2307.16176.pdf).

<img src="https://github.com/open-mmlab/mmdeploy/assets/110151316/6faad417-e5a9-45ed-98dd-2836c2a79aff" alt="teasor" style="zoom: 40%;" />



## Pretrained Model

You can download our pretrained model  [here](https://drive.google.com/file/d/1VLlwfsqNCrCzwhcOHOxXeG4jUXL_m7WU/view?usp=sharing).



## Testing

You can use `test.py` to test your model.

The model checkpoint should be placed in `pretrained/` . The default checkpoint name is `DHN_4channel.pth`, you can modify this by changing the value of `pretrainedModelDir` in `config.yml` .

The test image should be placed in `data/test/`. Three images are expected for model test:

- `cover.png` : The cover image for data embedding, usually a visualization image.
- `data_image.png` : A 3-channel image, each channel of which is a data image generated with our **Data-to-Image (DTOI)** algorithm.
- `qr_image.png` : A QR Code image containing one or more QR Codes encoded with chart information.

More details are presented in our paper.

We have prepared some images in `data/test/` for a quick start.

Once the aboved mentioned data is prepared, you can test your model with:

```bash
python test.py
```

The result images can be found in `result/` .



## Training

You can also train the model with your own data.

The training data should be ordered like:

```bash
data
|-- train
|   |-- MASSVIS  # or replace it with your own cover image dataset
|   |-- QR_Image_Dir # the directory of your QR Image dataset
|   |-- Data_Image_Dir1
|   |-- Data_Image_Dir2
|   |-- Data_Image_Dir3
	...
```

You can use more kinds of data image for training by modifying `dataloader.py` and `config.yml`.

Once the data is prepared, you can train your model with:

```bash
python train.py
```

The model checkpoints will be saved in `checkpoints/` .



## Citation

```bib
@InProceedings{VIS2023-InvVis,
  title={InvVis: Large-Scale Data Embedding for Invertible Visualization},
  author = {Huayuan Ye, Chenhui Li, Yang Li and Changbo Wang},
  booktitle={IEEE Transactions on Visualization and Computer Graphics},
  year={2023}
}
```



