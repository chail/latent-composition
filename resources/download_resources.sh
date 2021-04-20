### pretrained networks ###
# EDIT: pretrained networks now downloaded through torch.hub
# wget https://people.csail.mit.edu/lrchai/projects/latent_composition/pretrained_models.zip
# gdown --id 1vSEH2XMIG1XzQl3JLZwUKm_kdomelUqm
# unzip pretrained_models.zip

### external pretrained models ### 

# gan-seeing segmenter model
git clone https://github.com/davidbau/ganseeing.git resources/ganseeing

# saliency model
git clone https://github.com/Ugness/PiCANet-Implementation.git resources/PiCANet-Implementation
gdown --id 1Ga59ouyuEOpvRtiMvYHWqSoD313y4QD- # pretrained model
mv 36epo_383000step.ckpt resources/PiCANet-Implementation/

# face segmentation model
git clone https://github.com/zllrunning/face-parsing.PyTorch.git resources/face_parsing_pytorch
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812  # pretrained model
mkdir -p resources/face_parsing_pytorch/res/cp
mv 79999_iter.pth resources/face_parsing_pytorch/res/cp

# face landmarks model
mkdir -p resources/dlib
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat.bz2 resources/dlib
bunzip2 resources/dlib/shape_predictor_68_face_landmarks.dat.bz2


# identity loss model from pixel2style2pixel
gdown --id 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn # pretrained model
mkdir -p resources/psp
mv model_ir_se50.pth resources/psp
