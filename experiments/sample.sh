### generate samples from generator ### 

# celebahq dataset size = 30k
python -m experiments.sample --model proggan --domain celebahq \
	--seed 0 --num_samples 30000 --im_size 1024 \
	--outdir results/samples/{model}_{domain}_1024res_full

python -m experiments.sample --model proggan --domain livingroom \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/samples/{model}_{domain}_full

python -m experiments.sample --model proggan --domain church \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/samples/{model}_{domain}_full

python -m experiments.sample --model stylegan --domain church \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/samples/{model}_{domain}_full

python -m experiments.sample --model stylegan --domain car \
	--seed 0 --num_samples 50000 --im_size 512 \
	--outdir results/samples/{model}_{domain}_512res_full

python -m experiments.sample --model stylegan --domain ffhq \
	--seed 0 --num_samples 50000 --im_size 1024 \
	--outdir results/samples/{model}_{domain}_1024res_full

