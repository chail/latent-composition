# Reconstruction distances on the composites # 

for exproot in \
	results/composite_samples/stylegan_car_512res
do
for s in $(ls -d $exproot/inverted_*/)
do
	echo ======== $s ========
	python -m experiments.metric_distance_car ${s%%/}
done

done


