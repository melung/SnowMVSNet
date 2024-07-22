<h1 align="center">SnowMVSNet: Visibility-Aware Multi-View Stereo by Surface Normal Weighting for Occlusion Robustness.</h1>

<div align="center">
    <a href="https://github.com/melung" target='_blank'>Hyuksang Lee</a>, 
    <a href="" target='_blank'>Seongmin Lee</a>, 
    <a href="" target='_blank'>Sanghoon Lee</a>, 
</div>

<br />

## Setup
```
conda create -n snowmvs python=3.9
conda activate snowmvs
pip install -r requirements.txt
```

## Training

## DTU
* Dowload [DTU dataset](https://roboimagedata.compute.dtu.dk/). For convenience, can download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $DTU_TRAINING folder. For training and testing with raw image size, you can download [Rectified_raw](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip), and unzip it.

```                
├── Cameras    
├── Depths
├── Depths_raw   
├── Rectified
├── Rectified_raw (Optional)                                      
```
In ``scripts/train_dtu.sh``, set ``DTU_TRAINING`` as $DTU_TRAINING

Train SnowMVSNet: 
```
bash ./scripts/train_dtu.sh exp_name
```
After training, you will get model checkpoints in ./checkpoints/dtu/exp_name.

## Testing
* Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the $DTU_TESTPATH folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* In ``scripts/test_dtu.sh``, set ``DTU_TESTPATH`` as $DTU_TESTPATH.
* You can use my [pretrained model](https://drive.google.com/file/d/1bIgGtPT_aSCm_-DEExfQ1-ngoR1chyOI/view?usp=drive_link).

* Test:
```
bash ./scripts/test_dtu.sh exp_name
```
* Test with provided pretrained model:
```
bash scripts/test_dtu.sh pretrained --loadckpt PATH_TO_CKPT_FILE
```

## Citation
```



```


## Acknowledgements
Our work is partially baed on these opening source work: [MVSTER](https://github.com/JeffWang987/MVSTER), [GeoMvset](https://github.com/doubleZ0108/GeoMVSNet).

We appreciate their contributions to the MVS community.
