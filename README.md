# ViT-for-medical-image
project for Berkeley CS182/282A.

Fo run the code, make sure you have TensorDatasets named "train_dataset.pt" and "dev_dataset.pt" at the root dir (set to the parent dir). 

Follow the following commands:
    ./run_mae.sh
to run masked autoencoding task,

then, model will save the pre-trained ViT model at "root/mae_pretrained.pt"
run
    ./run_seg.sh
to run the segmentation task.
