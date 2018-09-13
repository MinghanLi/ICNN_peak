
The code includes three parts: CAM, ICNN, ICNN+peak.
The related papers can be found:
1. CAM:Learning Deep Features for Discriminative Localization
 http://medpapers.cn-hangzhou.oss.aliyun-inc.com/Learning%20Deep%20Features%20for%20Discriminative%20Localization.pdf
 
2. Interpretable Convolutional Neural Networks 
http://medpapers.cn-hangzhou.oss.aliyun-inc.com/Interpretable%20Convolutional%20Neural%20Networks.pdf

## MURA:

1. Download MURA dataset from OSS: oss://suqi000/
2. Replace cvs files in original dataset with train_Luna16_JPG.csv in ICNN_peak/2D/annotation
3. Run CAM/CAM_pai in Pytorch-Pai as follow:

  pai –name pytorch -Dscript='file:///Users/suqi.lmh/Musculoskeletal-Radiographs-Abnormality-Classifier/pytorch/CAM_py2/src.tar.gz' 
  -DentryFile='Train_CAM.py’ -Dvolumes='odps://asclepius/volumes/suqi/MURA_data,odps://asclepius/volumes/suqi/MURA_paths’ 
  -Dbucket='oss://suqi/CAM_baseline/' -Darn='acs:ram::1627427067181571:role/deepmedodps' -Dhost='oss-cn-hangzhou.aliyuncs.com' 
  -DworkerCount=4;

## LUNA16_2D:

1. Download Luna16_2D dataset from OSS: oss://tonggou000/
2. Replace cvs files in original dataset with MURA_paths in CAM/MURA_paths
3. Run main.py
