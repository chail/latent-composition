 # compositions on GAN samples

 python -m experiments.run_compositions --model proggan --domain church \
 	--seed 0 --num_samples 50000 --im_size 256 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}
 
 python -m experiments.run_compositions --model proggan --domain livingroom \
 	--seed 0 --num_samples 50000 --im_size 256 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}
 
 python -m experiments.run_compositions --model proggan --domain celebahq \
 	--seed 0 --num_samples 30000 --im_size 1024 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}_1024res

 python -m experiments.run_compositions --model stylegan --domain church \
 	--seed 0 --num_samples 50000 --im_size 256 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}
 
 python -m experiments.run_compositions --model stylegan --domain car \
 	--seed 0 --num_samples 50000 --im_size 512 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}_512res
 
 python -m experiments.run_compositions --model stylegan --domain ffhq \
 	--seed 0 --num_samples 50000 --im_size 1024 \
 	--input_source samples \
 	--outdir results/composite_{input_source}/{model}_{domain}_1024res
 
