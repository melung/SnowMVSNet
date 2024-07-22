<h1 align="center">SnowMVSNet: Visibility-Aware Multi-View Stereo by Surface Normal Weighting for Occlusion Robustness.</h1>

<div align="center">
    <a href="https://github.com/melung" target='_blank'>Hyuksang Lee</a>1, 
    <a href="" target='_blank'>Seongmin Lee</a>1, 
    <a href="http://insight.yonsei.ac.kr/gnuboard/bbs/content.php?co_id=member_prof" target='_blank'>Sanghoon Lee</a>, 
</div>

<br />

## Setup
```
conda create -n snowmvs python=3.9
conda activate snowmvs
pip install -r requirements.txt
```

## Training

##    DTU
* Download [DTU dataset](https://roboimagedata.compute.dtu.dk/) or [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (preprocessed by [MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it like bellow. If you want to train with raw image size, you can download [Rectified_raw](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip), and unzip it.

```                
├── Cameras    
├── Depths
├── Depths_raw   
├── Rectified
├── Rectified_raw                                
```

Train SnowMVSNet with DTU dataset: 
```
bash ./scripts/train_dtu.sh exp_name
```

##    BlendedMVS
* Download [low-res set](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) from [BlendedMVS](https://github.com/YoYo000/BlendedMVS) and unzip it like below:.

```                
├── dataset_low_res 
    ├── 5a3ca9cb270f0e3f14d0eddb      
    │    ├── blended_images
    │    ├── cams
    │    └── rendered_depth_maps
    ├── ...
    ├── all_list.txt
    ├── training_list.txt
    └── validation_list.txt                    
``` 

Train SnowMVSNet with BlendedMVS dataset: 
```
bash ./scripts/train_blend.sh exp_name
```


## Testing
##    DTU

* Download [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (preprocessed by [MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it.
* You can use my [pretrained model](https://drive.google.com/file/d/1bIgGtPT_aSCm_-DEExfQ1-ngoR1chyOI/view?usp=drive_link).

* Test:
```
bash ./scripts/test_dtu.sh exp_name
```
* Test with provided pretrained model:
```
bash scripts/test_dtu.sh pretrained --loadckpt PATH_TO_CKPT_FILE
```

* Pointcloud Fusion:
```
bash scripts/fusion_dtu.sh
```

##    Tanks and Temples

* Download [tank and temples data](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and unzip it.

* Test:
```
bash ./scripts/test_tnt.sh exp_name
```

* Pointcloud Fusion:
```
bash scripts/fusion_tnt.sh
```

##    MVHuman dataset

* Download [MVHuman dataset](TBD) and unzip it.

* Test:
```
bash ./scripts/test_mdi.sh exp_name
```

* Pointcloud Fusion:
```
TBD
```


## Citation
```



```


## Acknowledgements
Our work is partially baed on these opening source work: [MVSTER](https://github.com/JeffWang987/MVSTER), [GeoMvset](https://github.com/doubleZ0108/GeoMVSNet).

We appreciate their contributions to the MVS community.
