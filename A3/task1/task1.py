from ase.io import read
from gpaw import GPAW
from ase.md.npt import NPT
from ase.io import Trajectory

atoms = read('na_inserted.xyz')

calc = GPAW(mode='lcao',
            xc='PBE',
            basis='dzp',
            symmetry={'point_group': False}, # Turn off point-group symmetry
            charge=1, # Charged system
            txt='output.gpaw-out' # Redirects calculator output to this file
            )

atoms.set_calculator(calc)

from ase.units import fs, kB
dyn = NPT(atoms,
        temperature_K=350,
        timestep=0.5*fs, # Water scale timestep
        ttime=20*fs, # Dont forget the fs!
        pfactor=None,
        externalstress=0, # We dont use the barostat, but...
        logfile='mdOutput.log' # Outputs temperature (and more) to file at each timestep
        )

trajectory = Trajectory('someDynamics.traj', 'w', atoms)
dyn.attach(trajectory.write, interval=1) # Write the current positions etc. to file each timestep
dyn.run(4000) # Run 10 steps of MD simulation