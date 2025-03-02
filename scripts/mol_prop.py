from rdkit import Chem
from rdkit.Chem import Descriptors
import torch.utils
import torch.utils.data

import torch


def calculate_descriptions(smiles_batch):

    features = []

    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_weight = Descriptors.MolWt(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            mol_mr = Descriptors.MolMR(mol)
            formal_charge = Chem.rdmolops.GetFormalCharge(mol)
            fraction_csp3 = Descriptors.FractionCSP3(mol)
            features.append([mol_weight, num_h_donors,num_h_acceptors,
                             tpsa, mol_mr, formal_charge, fraction_csp3])

        else:
            features.append([0,0,0,0,0,0,0])

    return torch.tensor(features, dtype=torch.float32)

