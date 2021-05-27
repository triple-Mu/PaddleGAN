

# LapStyle


This repo holds the official codes of paper: "Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer", which is accepted in CVPR 2021.

## 1 Paper Introduction


Artistic style transfer aims at migrating the style from an example image to a content image. Currently, optimization- based methods have achieved great stylization quality, but expensive time cost restricts their practical applications. Meanwhile, feed-forward methods still fail to synthesize complex style, especially when holistic global and local patterns exist. Inspired by the common painting process ofdrawing a draft and revising the details, [this paper](https://arxiv.org/pdf/2104.05376.pdf)  introduce a novel feed- forward method Laplacian Pyramid Network (LapStyle). LapStyle first transfers global style pattern in low-resolution via a Drafting Network. It then revises the local details in high-resolution via a Revision Network, which hallucinates a residual image according to the draft and the image textures extracted by Laplacian filtering. Higher resolution details can be easily generated by stacking Revision Networks with multiple Laplacian pyramid levels. The final stylized image is obtained by aggregating outputs ofall pyramid levels. We also introduce a patch discriminator to better learn local pattern adversarially. Experiments demonstrate that our method can synthesize high quality stylized images in real time, where holistic style patterns are properly transferred.

![lapstyle_overview](https://user-images.githubusercontent.com/79366697/118654987-b24dc100-b81b-11eb-9430-d84630f80511.png)


## 2 How to use  

### 2.1 Prepare Datasets

To train LapStyle, we use the COCO dataset as content set. And you can choose any style image you like. Before training or testing, remember modify the data path of style image in the config file.


### 2.2 Train

Datasets used in example is COCO, you can also change it to your own dataset in the config file.

(1) Train the Draft Network of LapStyle under 128*128 resolution:
```
python -u tools/main.py --config-file configs/lapstyle_draft.yaml
```

(2) Then, train the Revision Network of LapStyle under 256*256 resolution:
```
python -u tools/main.py --config-file configs/lapstyle_rev_first.yaml --load ${PATH_OF_LAST_STAGE_WEIGHT}
```

(3) Further, you can train the second Revision Network under 512*512 resolution:

```
python -u tools/main.py --config-file configs/lapstyle_rev_second.yaml --load ${PATH_OF_LAST_STAGE_WEIGHT}
```

### 2.4 Test

To test the trained model, you can directly test the "lapstyle_rev_second", since it also contains the trained weight of previous stages:
```
python tools/main.py --config-file configs/lapstyle_rev_second.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 Results

| Style | Stylized Results |
| --- | --- |
| ![starrynew](https://user-images.githubusercontent.com/79366697/118655415-1ec8c000-b81c-11eb-8002-90bf8d477860.png) | ![chicago_stylized_starrynew](https://user-images.githubusercontent.com/79366697/118655671-59325d00-b81c-11eb-93a3-4fcc24680124.png)|
| ![ocean](https://user-images.githubusercontent.com/79366697/118655407-1c666600-b81c-11eb-83a6-300ee1952415.png) | ![chicago_ocean_512](https://user-images.githubusercontent.com/79366697/118655625-4cae0480-b81c-11eb-83ec-30936ed3df65.png)|
| ![stars](https://user-images.githubusercontent.com/79366697/118655423-20928380-b81c-11eb-92bd-0deeb320ff14.png) | ![chicago_stylized_stars_512](https://user-images.githubusercontent.com/79366697/118655638-50da2200-b81c-11eb-9223-58d5df022fa5.png)|
| ![circuit](https://user-images.githubusercontent.com/79366697/118655399-196b7580-b81c-11eb-8bc5-d5ece80c18ba.jpg) | ![chicago_stylized_circuit](https://user-images.githubusercontent.com/79366697/118655660-56376c80-b81c-11eb-87f2-64ae5a82375c.png)|

## 4 Pre-trained models

We also provide several trained models.

| model | style | path |
|---|---|---|
| lapstyle_circuit  | circuit | [lapstyle_circuit](https://paddlegan.bj.bcebos.com/models/lapstyle_circuit.pdparams)
| lapstyle_ocean  | ocean | [lapstyle_ocean](https://paddlegan.bj.bcebos.com/models/lapstyle_ocean.pdparams)
| lapstyle_starrynew  | starrynew | [lapstyle_starrynew](https://paddlegan.bj.bcebos.com/models/lapstyle_starrynew.pdparams)
| lapstyle_stars  | stars | [lapstyle_stars](https://paddlegan.bj.bcebos.com/models/lapstyle_stars.pdparams)


# References



```
@article{lin2021drafting,
  title={Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer},
  author={Lin, Tianwei and Ma, Zhuoqi and Li, Fu and He, Dongliang and Li, Xin and Ding, Errui and Wang, Nannan and Li, Jie and Gao, Xinbo},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```