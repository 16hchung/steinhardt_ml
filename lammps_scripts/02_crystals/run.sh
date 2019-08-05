lat=(fcc bcc hcp)
n=(6 8 6)

rm -f data/*
for i in `seq 0 2`; do
  echo ${lat[$i]}
  lmp_serial -in in.lmp -log data/lammps_${lat[$i]}.log -screen none -var RANDOM ${RANDOM} -var lat ${lat[$i]} -var n ${n[$i]} &
done
wait
