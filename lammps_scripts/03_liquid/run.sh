rm -f data/*
#lmp_serial -in in.lmp -log data/lammps_liquid.log -screen none -var RANDOM ${RANDOM}
lmp_serial -in in.lmp -log data/lammps_liquid.log -var RANDOM ${RANDOM}
