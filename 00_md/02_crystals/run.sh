#lat=(fcc bcc hcp)
lat=(fcc hd cd sc)
latlamp=(fcc hcp diamond sc)
#n=(6 8 6)
n=(10 10 9 15)
lmmps=perfect
#lmmps=real
suffix=_perfect
#suffix=

#rm -f data/*
for i in `seq 0 3`; do
  echo ${lat[$i]}
  #lmp_serial -in in.lmp -log data/logs/lammps_${lat[$i]}.log -screen none -var RANDOM ${RANDOM} -var lat ${lat[$i]} -var n ${n[$i]} &
  lmp_serial -in ${lmmps}.lmp -log new_data/log/lammps_${lat[$i]}${suffix}.log -screen none -var RANDOM ${RANDOM} -var lat ${latlamp[$i]} -var latlog ${lat[$i]} -var n ${n[$i]} -var sfx ${suffix} 
done
wait
