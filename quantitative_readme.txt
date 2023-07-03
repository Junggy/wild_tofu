quantitative.py will evaluate entire validation sequence per train_name, val_iter from both RGB, ToF branch.
As is designed to work with inferred depth from validation images obtained from training loop, it requires specific format for folder names.

i.e.)
"./results/{train_name}/inferece/{val_iter}/20201027-181125/rgb/*.png" <- for rgb depth
"./results/{train_name}/inferece/{val_iter}/20201027-181125/tof/*.png" <- for tof depth

Required parameters for running 

  1. -p : plot or not (True/False)
  2. -b : dataset folder directory (i.e. ../dataset)
  3. {train_name} {val_iter} pair (with space between). its possible to write multiple pair to compare with different iteration or different setup.

Example line :

[in]:
python quantitative.py -p False -b ../dataset full_fusion 10 full_fusion 20 full_fusion 30

(i.e. quantiative evaluation with "full_fusion" setup but with 10th 20th 30th validation iteration, and no plot.
      changing -p True will plot depth prediction per each frame for qualitative evaluation)

[out]: 
full_fusion val iter : 10
rmse rgb : 0.6052233040958817 rmse tof : 0.606849351863959
rel abs rgb : 0.07899654197996521 rel abs tof : 0.08085801122074211
rel sqr rgb : 0.07887840727193052 rel sqr tof : 0.09936222531701272
aut rgb : [93.47092279 98.11123146 99.19806558] aut tof : [93.33708729 98.08000082 99.16782119]

full_fusion val iter : 20
rmse rgb : 0.5882414366904135 rmse tof : 0.5854648336169807
rel abs rgb : 0.07613392236604684 rel abs tof : 0.07890062974725794
rel sqr rgb : 0.07271798278837444 rel sqr tof : 0.09414471533721017
aut rgb : [94.01951018 98.29063755 99.27283094] aut tof : [94.00400084 98.17520215 99.18263252]

full_fusion val iter : 30
rmse rgb : 0.5711534972305186 rmse tof : 0.5803981090287154
rel abs rgb : 0.0728406113639778 rel abs tof : 0.07699720908056244
rel sqr rgb : 0.06960955564680507 rel sqr tof : 0.09283157654201785
aut rgb : [94.31670121 98.38488482 99.33801869] aut tof : [94.21220189 98.22013061 99.21905097]


