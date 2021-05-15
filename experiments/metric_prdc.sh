source experiments/dataset_paths.sh
echo $dataset_celebahq

# ### PRDC on GAN samples ### 

python -m experiments.metric_prdc $dataset_church results/samples/proggan_church_full
python -m experiments.metric_prdc $dataset_livingroom results/samples/proggan_livingroom_full
python -m experiments.metric_prdc $dataset_celebahq results/samples/proggan_celebahq_1024res_full
python -m experiments.metric_prdc $dataset_church results/samples/stylegan_church_full
python -m experiments.metric_prdc $dataset_ffhq results/samples/stylegan_ffhq_full_1024res_full
python -m experiments.metric_prdc $dataset_car results/samples/stylegan_car_512res_full

### PRDC on encoder reconstructions ### 
python -m experiments.metric_prdc $dataset_church results/reencode/proggan_church_full
python -m experiments.metric_prdc $dataset_livingroom results/reencode/proggan_livingroom_full
python -m experiments.metric_prdc $dataset_celebahq results/reencode/proggan_celebahq_1024res_full
python -m experiments.metric_prdc $dataset_church results/reencode/stylegan_church_full
python -m experiments.metric_prdc $dataset_ffhq results/reencode/stylegan_ffhq_1024res_full
python -m experiments.metric_prdc $dataset_car results/reencode/stylegan_car_512res_full

### PRDC on the compositions ###
exproot=results/composite_samples/proggan_church
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_church ${s%%/}
done

exproot=results/composite_samples/proggan_livingroom
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_livingroom ${s%%/}
done

exproot=results/composite_samples/proggan_celebahq_1024res
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_celebahq ${s%%/}
done


exproot=results/composite_samples/stylegan_church
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_church ${s%%/}
done

exproot=results/composite_samples/stylegan_car_512res
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_car ${s%%/}
done

exproot=results/composite_samples/stylegan_ffhq_1024res
for s in $(ls -d $exproot/composite_*/) $(ls -d $exproot/inverted_*/) \
	$(ls -d $exproot/poisson/)
do
	echo ================ $s ==================
	python -m experiments.metric_prdc $dataset_ffhq ${s%%/}
done

