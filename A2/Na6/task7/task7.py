from ase import Atoms
from gpaw import GPAW, FermiDirac
from ase.io import write,read
from ase.optimize import GPMin
from pathlib import Path

Path("output/").mkdir(exist_ok=True)

Na6_low = read("christmas-tree.xyz")
Na6_high = read("half-decahedron.xyz")

fixed_params = {
    "nbands": 10,
    "occupations": FermiDirac(0.05),
    "setups": {'Na': '1'},
    "basis": "dzp"
}

params = [{
    "mode": "lcao",
    "xc": "LDA",
    "h": 0.25,
    "fmax": 0.2,
},
{
    "mode": "lcao",
    "xc": "LDA",
    "h": 0.15,
    "fmax": 0.05,
},
{
    "mode": "fd",
    "xc": "LDA",
    "h": 0.25,
    "fmax": 0.2,
},
{
    "mode": "pw",
    "xc": "LDA",
    "h": 0.25,
    "fmax": 0.2,
},
{
    "mode": "lcao",
    "xc": "LDA",
    "h": 0.15,
    "fmax": 0.2,
},
{
    "mode": "fd",
    "xc": "PBE",
    "h": 0.25,
    "fmax": 0.2,
},
{
    "mode": "fd",
    "xc": "PBE",
    "h": 0.25,
    "fmax": 0.05,
},
{
    "mode": "fd",
    "xc": "PBE",
    "h": 0.15,
    "fmax": 0.2,
},
{
    "mode": "fd",
    "xc": "PBE",
    "h": 0.15,
    "fmax": 0.05,
},
{
    "mode": "fd",
    "xc": "RPBE",
    "h": 0.25,
    "fmax": 0.2,
},]

for i in range(len(params)):
    temp_params = params[i].copy()
    fmax = temp_params.pop("fmax")

    Na6_low_copy = Na6_low.copy()
    Na6_high_copy = Na6_high.copy()

    calc = GPAW(**temp_params, **fixed_params, txt=f'output/out_{i:03d}Na6_low.txt')
    Na6_low_copy.set_calculator(calc)
    dyn = GPMin(Na6_low_copy, trajectory=f'output/{i:03d}Na6_low.traj', logfile=f'output/{i:03d}Na6_low.log')
    dyn.run(fmax=fmax, steps=100)
    
    calc = GPAW(**temp_params, **fixed_params, txt=f'output/out_{i:03d}Na6_high.txt')
    Na6_high_copy.set_calculator(calc)
    dyn = GPMin(Na6_high_copy, trajectory=f'output/{i:03d}Na6_high.traj', logfile=f'output/{i:03d}Na6_high.log')
    dyn.run(fmax=fmax, steps=100)

    params[i]["Na6_low_energy"] = Na6_low_copy.get_potential_energy()
    params[i]["Na6_high_energy"] = Na6_high_copy.get_potential_energy()

import pandas as pd
df = pd.DataFrame(params)
df.to_csv("output/results.csv")