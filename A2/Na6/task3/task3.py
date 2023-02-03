from ase import Atoms
from gpaw import GPAW, FermiDirac
from ase.io import write
from ase.optimize import GPMin

positions = [[1,0,0],
             [-1,0,0],
             [-0.5,3**0.5/2,0],
             [-0.5,-3**0.5/2,0],
             [0.5,3**0.5/2,0],
             [0.5,-3**0.5/2,0]]

cluster = Atoms("Na6",positions=positions)
a = 16.
cluster.set_cell((a,a,a))
cluster.center()

write("initial_guess.xyz",cluster)

calc = GPAW(nbands=10,
            h=0.25,
            txt='out.txt',
            occupations=FermiDirac(0.05),
            setups={'Na': '1'},
            mode='lcao',
            basis='dzp')

cluster.set_calculator(calc)
dyn = GPMin(cluster, trajectory='relax_ref.traj', logfile='relax_ref.log')
dyn.run(fmax=0.02, steps=100)

write("relaxed.xyz",cluster)
calc.write("relaxed.gpw")
