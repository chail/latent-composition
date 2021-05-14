mkdir -p training/checkpoints
mkdir -p training/runs

# train on fake images only

# ffhq encoder adds identity loss
python -m training.train_sgan_encoder \
	--batchSize 4 --netE_type resnet-34 \
	--netG ffhq --lr 0.0001 --niter 1500 \
	--lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 \
	--lambda_id 1.0 --masked \
	--outf training/checkpoints/sgan_encoder_{netG}_{netE_type}_RGBM

python -m training.train_sgan_encoder \
	--batchSize 16 --netE_type resnet-34 \
	--netG church --lr 0.0001 --niter 6800 \
	--lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 \
	--lambda_id 0.0 --masked \
	--outf training/checkpoints/sgan_encoder_{netG}_{netE_type}_RGBM

python -m training.train_sgan_encoder \
	--batchSize 16 --netE_type resnet-34 \
	--netG horse --lr 0.0001 --niter 6800 \
	--lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 \
	--lambda_id 0.0 --masked \
	--outf training/checkpoints/sgan_encoder_{netG}_{netE_type}_RGBM

python -m training.train_sgan_encoder \
	--batchSize 8 --netE_type resnet-34 \
	--netG car --lr 0.0001 --niter 6800 \
	--lambda_mse 1.0 --lambda_lpips 1.0 --lambda_latent 1.0 \
	--lambda_id 0.0 --masked \
	--outf training/checkpoints/sgan_encoder_{netG}_{netE_type}_RGBM
