rm -f data/*
echo "# Lattice | Density | Energy" > data/equilibrium_density.dat
for lat in 'fcc' 'bcc' 'hcp'; do
  lmp_serial -in in.lmp -screen none -log data/log/lammps_${lat}.log -var lat ${lat} 
done
