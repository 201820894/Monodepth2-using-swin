# Monodepth2

============================================================================================================================
This is Pytorch implementation of MonodepthV2 using swin transformerV2

This repo is based on [Swin transformer](https://github.com/microsoft/Swin-Transformer) by [ChristophReich1996](https://github.com/ChristophReich1996/Swin-Transformer-V2) and [MonodepthV2](https://github.com/nianticlabs/monodepth2)

All setup and dataset follows [MonodepthV2](https://github.com/nianticlabs/monodepth2).

We train [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip) model as scratch and compared to original MonodepthV2 using ResNet encoder

## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>/weights_<epoch>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model --png
```

python test_simple_swin.py --image_path assets/school_test.jpg --model_name weights_19

## 🖼️ Prediction for a single image

**Single Image prediction:**
```shell
python test_simple_swin.py --image_path <img_path> --model_name <model name>
```

## 📊 KITTI evaluation

**Evaluation:**
```shell
python evaluate_depth.py --load_weights_folder <load weight folder> --eval_mono --png
```


| `--model_name`          | Training modality | Imagenet pretrained? | Model resolution  | KITTI abs. rel. error |  delta < 1.25  |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|
| [`mono_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip)          | Mono              | No | 640 x 192                | 0.132                 | 0.845          |
| [`Swin_no_pt_640x192`](https://drive.google.com/file/d/14SPJSkah0AUrIju_M_tf_1MD2AakEYdm/view?usp=share_link)        | Mono            | No | 640 x 192                | 0.126                 | 0.848          |
