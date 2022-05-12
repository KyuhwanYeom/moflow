import argparse
import os
import sys
# for linux env.
sys.path.insert(0,'..')
from distutils.util import strtobool

import pickle
import torch

import numpy as np

from data.data_loader import NumpyTupleDataset
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
# from mflow.generate import generate_mols_along_axis
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_validity, construct_mol, adj_to_smiles
from mflow.utils.model_utils import load_model, get_latent_vec, smiles_to_adj
from mflow.utils.molecular_metrics import MolecularMetrics
from mflow.models.model import MoFlow, rescale_adj
from mflow.utils.timereport import TimeReport
import mflow.utils.environment as env
from sklearn.linear_model import LinearRegression
import time
import functools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_dir)
    smiles = df['smile_new'].values
    smiles_list = smiles.tolist()
    
    sim = []
    for i in smiles_list:
        for j in smiles_list:
            if i!=j:
                mol1 = Chem.MolFromSmiles(i)
                mol2 = Chem.MolFromSmiles(j)
                fp1 = AllChem.GetMorganFingerprint(mol1, 2)
                fp2 = AllChem.GetMorganFingerprint(mol2, 2)
                sim.append(DataStructs.TanimotoSimilarity(fp1, fp2))
    print(np.mean(sim))