### reencode samples from G --> GEG(z) ###

python -m experiments.reencode --model proggan --domain livingroom \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/reencode/{model}_{domain}_full

python -m experiments.reencode --model proggan --domain church \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/reencode/{model}_{domain}_full

# celebahq dataset size = 30k
python -m experiments.reencode --model proggan --domain celebahq \
      --seed 0 --num_samples 30000 --im_size 1024 \
      --outdir results/reencode/{model}_{domain}_1024res_full

python -m experiments.reencode --model stylegan --domain church \
	--seed 0 --num_samples 50000 --im_size 256 \
	--outdir results/reencode/{model}_{domain}_full

python -m experiments.reencode --model stylegan --domain ffhq \
	--seed 0 --num_samples 50000 --im_size 1024 \
	--outdir results/reencode/{model}_{domain}_1024res_full

python -m experiments.reencode --model stylegan --domain car \
	--seed 0 --num_samples 50000 --im_size 512 \
	--outdir results/reencode/{model}_{domain}_512res_full

