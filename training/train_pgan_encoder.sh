mkdir -p training/checkpoints
mkdir -p training/runs

# celebahq RGBM
python -m training.train_pgan_encoder \
	--netE_type resnet-18 --niter 1000 --lr 0.0001 --batchSize 4 \
	--netG celebahq --lambda_mse 1.0 --lambda_lpips 1.0 \
	--masked --outf training/checkpoints/pgan_encoder_{netG}_{netE_type}_RGBM

# church RGBM
python -m training.train_pgan_encoder \
	--netE_type resnet-18 --niter 5000 --lr 0.0001 --batchSize 16 \
	--netG church --lambda_mse 1.0 --lambda_lpips 1.0 \
	--masked --outf training/checkpoints/pgan_encoder_{netG}_{netE_type}_RGBM

# living room RGBM
python -m training.train_pgan_encoder \
	--netE_type resnet-18 --niter 5000 --lr 0.0001 --batchSize 16 \
	--netG livingroom --lambda_mse 1.0 --lambda_lpips 1.0 \
	--masked --outf training/checkpoints/pgan_encoder_{netG}_{netE_type}_RGBM
