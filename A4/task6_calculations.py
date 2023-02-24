# %%
import json

metals = ["Au","Pt","Rh"]
adsorbates = ["O", "CO"]

with open("results.json","r") as json_file:
    r = json.load(json_file)

for metal in metals:
    for adsorbate in adsorbates:
        E_adsorbant = r[f"E_slab_{metal}"]

        if adsorbate == "O":
            E_adsorbate = r["E_mol_O2"]/2
        elif adsorbate == "CO":
            E_adsorbate = r["E_mol_CO"]

        E_combined = r[f"E_combined_{metal}_{adsorbate}"]

        E_ads = E_adsorbate + E_adsorbant - E_combined
        
        r[f"E_ads_{metal}_{adsorbate}"] = E_ads
        print(f"E_ads_{metal}_{adsorbate} = {E_ads:.3f} eV")

    E_ads_O = r[f"E_ads_{metal}_O"]
    E_ads_CO = r[f"E_ads_{metal}_CO"]
    E_a = -0.3*(E_ads_O+E_ads_CO)+0.22

    r[f"E_a_{metal}"] = E_a
    print(f"E_a_{metal} = {E_a:.3f} eV")


with open("results.json",'w') as json_file:
    json.dump(r, json_file, indent=4)
# %%
