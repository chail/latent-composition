# note: update paths to corresponding dataset directory

# 30k celebahq images
dataset_celebahq=resources/datasets/celebahq1024_30k

# 50K lsun church images
dataset_church=resources/datasets/church_outdoor_train_50k

# 50K lsun living room images
dataset_livingroom=resources/datasets/living_room_train_50k

# 50K ffhq images
dataset_ffhq=resources/datasets/ffhq1024

# 50k lsun car images. Note: they need to prepared accordingly to
# the GAN's aspect ratio in order to get similar FID as reported
# numbers in stylegan paper, which adds black padding to top and 
# bottom; see: 
# https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py#L461-L472
dataset_car=resources/datasets/cars512_50k

