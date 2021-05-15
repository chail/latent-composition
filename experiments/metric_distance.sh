# Reconstruction distances on the composites - between
# collage input and generator output

for exproot in \
	results/composite_samples/proggan_church \
	results/composite_samples/proggan_livingroom \
	results/composite_samples/proggan_celebahq_1024res \
	results/composite_samples/stylegan_church \
	results/composite_samples/stylegan_ffhq_1024res \
do

for s in $(ls -d $exproot/inverted_*/)
do
	echo ======== $s ========
	python -m experiments.metric_distance ${s%%/}
done

done

# removes the padding in the car images before computing similarity
for exproot in \
	results/composite_samples/stylegan_car_512res
do

for s in $(ls -d $exproot/inverted_*/)
do
	echo ======== $s ========
	python -m experiments.metric_distance ${s%%/} --crop_aspect_car
done
done
