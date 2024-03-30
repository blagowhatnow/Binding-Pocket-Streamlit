#Made wth streamlit 1.3.1
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA
import matplotlib.pyplot as plt
import SessionState
from rdkit import rdBase
from sascore import SAscore
from joblib import load
import molfil
import subprocess
import math
from multiprocessing import Process


def filter_valid_smiles(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Check validity of SMILES in each row
    valid_smiles = []
    for smiles in df["Ligand SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
    # Filter the DataFrame to include only rows with valid SMILES
    filtered_df = df[df["Ligand SMILES"].isin(valid_smiles)]
    return filtered_df

def run_chemprop_checkpoint(checkpoint_dir, preds_name):
    cmd = f"chemprop_predict --test_path test_data.csv --checkpoint_dir {checkpoint_dir} --preds_path {preds_name} --features_generator rdkit_2d_normalized --no_features_scaling"
    subprocess.run(cmd, shell=True)

def create_adme_dataframe(dframe):
    smiles = dframe['Ligand SMILES']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)

    checkpoints = [
        ("/home/beda/admet-master/ADME/absorption/Caco2_Wang/checkpoints_features", "caco"),
        ("/home/beda/admet-master/ADME/absorption/HIA_Hou/checkpoints_features", "hia_hou"),
        ("/home/beda/admet-master/ADME/absorption/PAMPA_NCATS/checkpoints_features", "pampa"),
        ("/home/beda/admet-master/ADME/absorption/Pgp_Broccatelli/checkpoints_features", "pgp"),
        ("/home/beda/admet-master/ADME/absorption/Solubility_AqSolDB/checkpoints_features", "solubility"),
        ("/home/beda/admet-master/ADME/absorption/Lipophilicity_AstraZeneca/checkpoints_features", "lipo"),
        ("/home/beda/admet-master/ADME/distribution/BBB_Martins", "bbb"),
        ("/home/beda/admet-master/ADME/metabolism/CYP1A2_Veith/checkpoints_features", "CYP1A2_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C19_Veith/checkpoints_features", "CYP2C19_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Substrate_CarbonMangels/checkpoints_features", "CYP2C9_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Veith/checkpoints_features", "CYP2C9_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Substrate_CarbonMangels/checkpoints_features", "CYP2D6_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Veith/checkpoints_features", "CYP2D6_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Substrate_CarbonMangels/checkpoints_features", "CYP3A4_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Veith/checkpoints_features", "CYP3A4_in"),
        ("/home/beda/admet-master/ADME/excretion/Clearance_Hepatocyte_AZ/checkpoints_features", "clear_hep"),
        ("/home/beda/admet-master/ADME/excretion/Half_Life_Obach_log/checkpoints_features", "hl"),
        ("/home/beda/admet-master/ADME/distribution/VDss_Lombardo_log/checkpoints_features","vdss"),
        ("/home/beda/admet-master/ADME/distribution/PPBR_AZ/checkpoints_features","ppbr")

    ]


    processes = []
    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff['Y']
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)

    subprocess.run(["rm test_data.csv"], shell=True)
    final_df['FDA_Drug_Names']=dframe['FDA drugnames']
    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    struc_images = [Draw.MolToImage(mol, size=(300, 300)) for mol in molecules]
    # Create a DataFrame with structure images
    struc_df = pd.DataFrame({'Structures': struc_images})
    merged_df_final = pd.concat([final_df, struc_df], axis=1)
    return merged_df_final

def create_adme_dataframe_single(dframe):
    smiles = dframe['smiles']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)


    checkpoints = [
        ("/home/beda/admet-master/ADME/absorption/Caco2_Wang/checkpoints_features", "caco"),
        ("/home/beda/admet-master/ADME/absorption/HIA_Hou/checkpoints_features", "hia_hou"),
        ("/home/beda/admet-master/ADME/absorption/PAMPA_NCATS/checkpoints_features", "pampa"),
        ("/home/beda/admet-master/ADME/absorption/Pgp_Broccatelli/checkpoints_features", "pgp"),
        ("/home/beda/admet-master/ADME/absorption/Solubility_AqSolDB/checkpoints_features", "solubility"),
        ("/home/beda/admet-master/ADME/absorption/Lipophilicity_AstraZeneca/checkpoints_features", "lipo"),
        ("/home/beda/admet-master/ADME/distribution/BBB_Martins", "bbb"),
        ("/home/beda/admet-master/ADME/metabolism/CYP1A2_Veith/checkpoints_features", "CYP1A2_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C19_Veith/checkpoints_features", "CYP2C19_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Substrate_CarbonMangels/checkpoints_features", "CYP2C9_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Veith/checkpoints_features", "CYP2C9_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Substrate_CarbonMangels/checkpoints_features", "CYP2D6_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Veith/checkpoints_features", "CYP2D6_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Substrate_CarbonMangels/checkpoints_features", "CYP3A4_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Veith/checkpoints_features", "CYP3A4_in"),
        ("/home/beda/admet-master/ADME/excretion/Clearance_Hepatocyte_AZ/checkpoints_features", "clear_hep"),
        ("/home/beda/admet-master/ADME/excretion/Half_Life_Obach_log/checkpoints_features", "hl"),
        ("/home/beda/admet-master/ADME/distribution/VDss_Lombardo_log/checkpoints_features","vdss"),
        ("/home/beda/admet-master/ADME/distribution/PPBR_AZ/checkpoints_features","ppbr")

    ]
    processes = []
   

    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff['Y']
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)

    subprocess.run(["rm test_data.csv"], shell=True)
    return final_df

def create_tox_dataframe(dframe):
    smiles = dframe['Ligand SMILES']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)

    checkpoints = [
        ("/home/beda/admet-master/toxicity/clintox/checkpoints_features", "clintox"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/cardio/checkpoints_features", "cardio"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/hepato/checkpoints_features", "hepato"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/respiratory/checkpoints_features", "respiratory"),
        ("/home/beda/admet-master/toxicity/cytotoxicity/checkpoints_features", "cytotoxicity"),
        ("/home/beda/admet-master/toxicity/mutagenicity/checkpoints_features", "mutagen"),
        ("/home/beda/admet-master/toxicity/carcinogenecity/checkpoints_features", "carci"),
        ("/home/beda/admet-master/toxicity/LD50/Mouse_oral/checkpoints_features", "mouse_oral")
    ]

    processes = []
    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff.iloc[:,1]
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)
    final_df['FDA_Drug_Names']=dframe['FDA drugnames']
    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    struc_images = [Draw.MolToImage(mol, size=(300, 300)) for mol in molecules]
    # Create a DataFrame with structure images
    struc_df = pd.DataFrame({'Structures': struc_images})
    merged_df_final = pd.concat([final_df, struc_df], axis=1)
    return merged_df_final

def create_adme_dataframe_ana(dframe):
    smiles = dframe['Ligand SMILES']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)

    checkpoints = [
        ("/home/beda/admet-master/ADME/absorption/Caco2_Wang/checkpoints_features", "caco"),
        ("/home/beda/admet-master/ADME/absorption/HIA_Hou/checkpoints_features", "hia_hou"),
        ("/home/beda/admet-master/ADME/absorption/PAMPA_NCATS/checkpoints_features", "pampa"),
        ("/home/beda/admet-master/ADME/absorption/Pgp_Broccatelli/checkpoints_features", "pgp"),
        ("/home/beda/admet-master/ADME/absorption/Solubility_AqSolDB/checkpoints_features", "solubility"),
        ("/home/beda/admet-master/ADME/absorption/Lipophilicity_AstraZeneca/checkpoints_features", "lipo"),
        ("/home/beda/admet-master/ADME/distribution/BBB_Martins", "bbb"),
        ("/home/beda/admet-master/ADME/metabolism/CYP1A2_Veith/checkpoints_features", "CYP1A2_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C19_Veith/checkpoints_features", "CYP2C19_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Substrate_CarbonMangels/checkpoints_features", "CYP2C9_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2C9_Veith/checkpoints_features", "CYP2C9_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Substrate_CarbonMangels/checkpoints_features", "CYP2D6_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP2D6_Veith/checkpoints_features", "CYP2D6_in"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Substrate_CarbonMangels/checkpoints_features", "CYP3A4_sub"),
        ("/home/beda/admet-master/ADME/metabolism/CYP3A4_Veith/checkpoints_features", "CYP3A4_in"),
        ("/home/beda/admet-master/ADME/excretion/Clearance_Hepatocyte_AZ/checkpoints_features", "clear_hep"),
        ("/home/beda/admet-master/ADME/excretion/Half_Life_Obach_log/checkpoints_features", "hl"),
        ("/home/beda/admet-master/ADME/distribution/VDss_Lombardo_log/checkpoints_features","vdss"),
        ("/home/beda/admet-master/ADME/distribution/PPBR_AZ/checkpoints_features","ppbr")

    ]

    processes = []
    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff['Y']
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)

    subprocess.run(["rm test_data.csv"], shell=True)
    return final_df

def create_tox_dataframe_ana(dframe):
    smiles = dframe['Ligand SMILES']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)

    checkpoints = [
        ("/home/beda/admet-master/toxicity/clintox/checkpoints_features", "clintox"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/cardio/checkpoints_features", "cardio"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/hepato/checkpoints_features", "hepato"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/respiratory/checkpoints_features", "respiratory"),
        ("/home/beda/admet-master/toxicity/cytotoxicity/checkpoints_features", "cytotoxicity"),
        ("/home/beda/admet-master/toxicity/mutagenicity/checkpoints_features", "mutagen"),
        ("/home/beda/admet-master/toxicity/carcinogenecity/checkpoints_features", "carci"),
        ("/home/beda/admet-master/toxicity/LD50/Mouse_oral/checkpoints_features", "mouse_oral")
    ]

    processes = []
    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff.iloc[:,1]
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)
    return final_df


def create_tox_dataframe_single(dframe):
    smiles = dframe['smiles']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)

    checkpoints = [
        ("/home/beda/admet-master/toxicity/clintox/checkpoints_features", "clintox"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/cardio/checkpoints_features", "cardio"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/hepato/checkpoints_features", "hepato"),
        ("/home/beda/admet-master/toxicity/organ_toxicity/respiratory/checkpoints_features", "respiratory"),
        ("/home/beda/admet-master/toxicity/cytotoxicity/checkpoints_features", "cytotoxicity"),
        ("/home/beda/admet-master/toxicity/mutagenicity/checkpoints_features", "mutagen"),
        ("/home/beda/admet-master/toxicity/carcinogenecity/checkpoints_features", "carci"),
        ("/home/beda/admet-master/toxicity/LD50/Mouse_oral/checkpoints_features", "mouse_oral")
    ]

    processes = []
    for checkpoint_dir, preds_name in checkpoints:
        process = Process(target=run_chemprop_checkpoint, args=(checkpoint_dir, f"{preds_name}_checking.csv"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    final_df = pd.DataFrame()
    for _, preds_name in checkpoints:
        dff = pd.read_csv(f"{preds_name}_checking.csv")
        final_df[preds_name] = dff.iloc[:,1]
        final_df[preds_name] = final_df[preds_name].apply(lambda x: round(x, 2))
        subprocess.run([f"rm {preds_name}_checking.csv"], shell=True)
    return final_df

def load_data(smiles_list):
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    X = []
    cnt = 0
    for mol in mols:
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_string = fp.ToBitString()
        tmpX = np.array(list(fp_string),dtype=float)
        X.append(tmpX)
        cnt += 1
    X = np.array(X)
    return X, smiles_list

def predict(smiles_list, model_file):
    df = pd.DataFrame(columns=['smiles', 'Tox-score', 'SAscore'])
    # laod the data
    X, smiles_list = load_data(smiles_list)
    # load the saved model and make predictions
    clf = load(model_file)
    reg = SAscore()
    for i in range(X.shape[0]):
        tox_score = clf.predict_proba(X[i,:].reshape((1,1024)))[:,1]
        sa_score = reg(smiles_list[i])
        df.at[i, 'smiles'] = smiles_list[i]
        df.at[i, 'Tox-score'] = tox_score[0]
        df.at[i, 'SAscore'] = sa_score
    return df

def filter_valid_smiles(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check validity of SMILES in each row
    valid_smiles = []
    for smiles in df["Ligand SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)

    # Filter the DataFrame to include only rows with valid SMILES
    filtered_df = df[df["Ligand SMILES"].isin(valid_smiles)].reset_index(drop=True)

    return filtered_df

def calculate_sasa(mol):
    try:
        hmol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(hmol)
        radii = rdFreeSASA.classifyAtoms(hmol)
        sasa = rdFreeSASA.CalcSASA(hmol, radii)
    except:
        sasa = float('nan')
    return sasa

def calculate_qed(mol):
    qed = Descriptors.qed(mol)
    return qed

def generate(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    properties = []
    for mol in moldata:
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)

        desc_MolLogP = round(Descriptors.MolLogP(mol), 3)
        desc_MolWt = round(Descriptors.MolWt(mol), 3)
        desc_NumRotatableBonds = round(Descriptors.NumRotatableBonds(mol), 3)
        desc_HBONDAcceptors = round(Chem.Lipinski.NumHDonors(mol), 3)
        desc_HBONDDonors = round(Chem.Lipinski.NumHAcceptors(mol), 3)
        desc_TPSA = round(Descriptors.TPSA(mol), 3)
        desc_HeavyAtoms = round(Descriptors.HeavyAtomCount(mol), 3)
        desc_NumAromaticRings = round(Descriptors.NumAromaticRings(mol), 3)
        desc_QED = round(calculate_qed(mol), 3)
        desc_SASA = round(calculate_sasa(mol), 3)

        properties.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds,
                           desc_HBONDAcceptors, desc_HBONDDonors, desc_TPSA,
                           desc_HeavyAtoms, desc_NumAromaticRings, desc_SASA, desc_QED])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                   "TPSA", "HeavyAtoms", "NumAromaticRings", "SASA", "QED"]

    descriptors = pd.DataFrame(data=properties, columns=columnNames)

    return descriptors

#No SASA
def generate_copy(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    properties = []
    for mol in moldata:
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)

        desc_MolLogP = round(Descriptors.MolLogP(mol), 3)
        desc_MolWt = round(Descriptors.MolWt(mol), 3)
        desc_NumRotatableBonds = round(Descriptors.NumRotatableBonds(mol), 3)
        desc_HBONDAcceptors = round(Chem.Lipinski.NumHDonors(mol), 3)
        desc_HBONDDonors = round(Chem.Lipinski.NumHAcceptors(mol), 3)
        desc_TPSA = round(Descriptors.TPSA(mol), 3)
        desc_HeavyAtoms = round(Descriptors.HeavyAtomCount(mol), 3)
        desc_NumAromaticRings = round(Descriptors.NumAromaticRings(mol), 3)
        desc_QED = round(calculate_qed(mol), 3)

        properties.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds,
                           desc_HBONDAcceptors, desc_HBONDDonors, desc_TPSA,
                           desc_HeavyAtoms, desc_NumAromaticRings, desc_QED])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                   "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

    descriptors = pd.DataFrame(data=properties, columns=columnNames)

    return descriptors

def generate_single(smiles, model_file):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles_list=[smiles]
        df_toxi=predict(smiles_list,model_file)

        if mol is None:
            return None, {}

        AllChem.EmbedMolecule(mol)

        desc_MolLogP = round(Descriptors.MolLogP(mol), 3)
        desc_MolWt = round(Descriptors.MolWt(mol), 3)
        desc_NumRotatableBonds = round(Descriptors.NumRotatableBonds(mol), 3)
        desc_HBONDAcceptors = round(Chem.Lipinski.NumHDonors(mol), 3)
        desc_HBONDDonors = round(Chem.Lipinski.NumHAcceptors(mol), 3)
        desc_TPSA = round(Descriptors.TPSA(mol), 3)
        desc_HeavyAtoms = round(Descriptors.HeavyAtomCount(mol), 3)
        desc_NumAromaticRings = round(Descriptors.NumAromaticRings(mol), 3)
        desc_QED = round(calculate_qed(mol), 3)
        desc_SASA = round(calculate_sasa(mol), 3)

        properties = {
            "MolLogP": desc_MolLogP,
            "MolWt": desc_MolWt,
            "NumRotatableBonds": desc_NumRotatableBonds,
            "HBondDonors": desc_HBONDDonors,
            "HBondAcceptors": desc_HBONDAcceptors,
            "TPSA": desc_TPSA,
            "HeavyAtoms": desc_HeavyAtoms,
            "NumAromaticRings": desc_NumAromaticRings,
            "SASA": desc_SASA,
            "QED": desc_QED,
            "Toxicity" : round(df_toxi['Tox-score'][0],3),
            "SA Score" : round(df_toxi['SAscore'][0],3)
        }

        return mol, properties
    except:
        return None, {}

def ret_final_df(df, model_file):
    df_orig = df.copy()  # Make a copy of the original DataFrame
    smiles = df_orig['Ligand SMILES']
    df = generate(smiles)
    df_tox=predict(smiles, model_file)
    df_sub_0=df_tox[['Tox-score', 'SAscore']]
    df_sub = df_orig[['FDA drugnames', 'Ligand SMILES']]
    merged_df = pd.concat([df_sub, df], axis=1)
    # Create RDKit molecules from the SMILES
    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    # Read the SDF file and extract the structures as images
    struc_images = [Draw.MolToImage(mol, size=(300, 300)) for mol in molecules]
    # Create a DataFrame with structure images
    struc_df = pd.DataFrame({'Structures': struc_images})
    # Merge the structure images DataFrame with the properties DataFrame
    merged_df_final = pd.concat([merged_df, struc_df], axis=1)
    merged_df_fin=pd.concat([df_sub_0, merged_df_final], axis=1)
    return merged_df_fin

def filter_dataframe(final_df2, num_pass):
    optimal_ranges = {
        'CYP2C9_sub': (0, 0.45),
        'cytotoxicity': (0, 0.59),
        'pgp': (0, 0.96),
        'caco': (-6.20, None),
        'carci': (0, 0.73),
        'mouse_oral': (-3.42, None),
        'CYP3A4_sub': (0, 0.86),
        'clintox': (0, 0.54),
        'CYP2D6_sub': (0, 0.65),
        'pampa': (0.15, None),
        'respiratory': (0, 0.97),
        'solubility': (-6.57, 0.57),
        'CYP3A4_in': (0, 0.95),
        'CYP2D6_in': (0, 0.88),
        'mutagen': (0, 0.89),
        'hl': (0.32, None),
        'cardio': (0, 0.91),
        'CYP1A2_in': (0, 0.93),
        'lipo': (-2.70, 4.38),
        'ppbr': (31.13, None),
        'hepato': (0, 0.97),
        'CYP2C19_in': (0, 0.82),
        'NR-AR': (0, 0.13),
        'NR-AR-LBD': (0, 0.12),
        'NR-AhR': (0, 0.36),
        'NR-Aromatase': (0, 0.26),
        'NR-ER': (0, 0.29),
        'NR-ER-LBD': (0, 0.10),
        'NR-PPAR-gamma': (0, 0.08),
        'SR-ARE': (0, 0.64),
        'SR-ATAD5': (0, 0.09),
        'SR-HSE': (0, 0.13),
        'SR-MMP': (0, 0.55),
        'SR-p53': (0, 0.27),
        'CYP2C9_in': (0, 0.80),
        'bbb': (0.50, None)
    }
    

    filtered_df = final_df2.copy()
    filtered_df['Filters_Passed'] = 0

    for column, (min_value, max_value) in optimal_ranges.items():
        min_condition = (min_value is not None)
        max_condition = (max_value is not None)

        if min_condition and max_condition:
            filtered_df['Filters_Passed'] += (filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)
        elif min_condition:
            filtered_df['Filters_Passed'] += (filtered_df[column] >= min_value)
        elif max_condition:
            filtered_df['Filters_Passed'] += (filtered_df[column] <= max_value)
    # Filter based on additional criteria and count them
    smiles = final_df2['Ligand SMILES']
    out = molfil.rule.filter(smiles)
    filter_count = []
    for item in out:
    # Count the number of False values in properties (excluding 'smiles')
       filters_passed = sum(1 for key, value in item.items() if key != 'smiles' and value is False)
       filter_count.append(filters_passed)
    filtered_df['Additional_Filters_Passed'] = filter_count
    # Count the total number of filters passed (admet + additional)
    filtered_df['Total_Filters_Passed'] = filtered_df['Filters_Passed'] + filtered_df['Additional_Filters_Passed']
    filtered_df= filtered_df.drop(['Filters_Passed','Additional_Filters_Passed'], axis=1)
    filtered_df=filtered_df[filtered_df['Total_Filters_Passed']>=num_pass]
    return filtered_df

def molecular_properties():
    model_file='etoxpred_best_model.joblib'
    st.title("Molecular Properties App - Knowdis")

    # Input field for molecule
    molecule_input = st.text_input("Enter a SMILES string:")

    if molecule_input:
        mol, properties = generate_single(molecule_input, model_file)

        if mol is None:
            st.write("Invalid molecule.")

        else:
            Chem.AllChem.Compute2DCoords(mol)
            # Display molecule properties
            st.subheader("Molecule Properties")
            st.write("MolLogP:", properties["MolLogP"])
            st.write("MolWt:", properties["MolWt"])
            st.write("NumRotatableBonds:", properties["NumRotatableBonds"])
            st.write("HBondDonors:", properties["HBondDonors"])
            st.write("HBondAcceptors:", properties["HBondAcceptors"])
            st.write("TPSA:", properties["TPSA"])
            st.write("HeavyAtoms:", properties["HeavyAtoms"])
            st.write("NumAromaticRings:", properties["NumAromaticRings"])
            st.write("SASA:", properties["SASA"])
            st.write("QED:", properties["QED"])
            st.write("Toxicity:", properties["Toxicity"])
            st.write("SA Score:", properties["SA Score"])

            # Display 2D structure
            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol, size=(300, 300)), use_column_width=False, width=300)
     # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")
    st.write("Results will keep only the valid SMILES strings from the uploaded dataframe. Default dataframe will be shown if no file is uploaded.")                   
    if uploaded_file is not None:
      try:
        # Read the CSV file into a DataFrame
          df=filter_valid_smiles(uploaded_file)
          N = 12
          session_state = SessionState.get(page_number = 0)
          molecules_per_row = 3 
          last_page = len(df) // N
          prev, _ ,next = st.columns([1, 10, 1])
          if next.button("Next"):
              if session_state.page_number + 1 > last_page:
                  session_state.page_number = 0
              else:
                  session_state.page_number += 1
          if prev.button("Previous"):
              if session_state.page_number - 1 < 0:
                  session_state.page_number = last_page
              else:
                  session_state.page_number -= 1
          start = session_state.page_number * N 
          end = (1 + session_state.page_number) * N

          if session_state.page_number==0:
                df_graphs=generate_copy(df['Ligand SMILES'])
                non_empty_figures = []
                properties = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                      "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
                axes = axes.flatten()

                for i, prop in enumerate(properties):
                   ax = axes[i]
                   _, _, patches = ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                   ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                   ax.set_title(prop, fontsize=25)
                   ax.set_xlabel("Value",fontsize=22 )
                   ax.set_ylabel("Frequency", fontsize=22)
                   ax.tick_params(axis='x', labelsize=18)
                   ax.tick_params(axis='y', labelsize=18)
               # Check if the figure is empty
                   if any(patches):
                      non_empty_figures.append(ax)
# Check if there are non-empty figures
                if non_empty_figures:
    # Display the non-empty figures
                    fig.subplots_adjust(hspace=0.5)
                    st.pyplot(fig)
                else:
                    pass
       # Slice the DataFrame based on the current page
          sliced_df = df[start:end].reset_index(drop=True)

        # Display the sliced DataFrame
          st.subheader("Original DataFrame Head")
          st.dataframe(sliced_df)


        # Process the DataFrame
          final_df = ret_final_df(sliced_df, model_file)

        # Display the structures with properties for the current page
          for i in range(0, len(final_df), molecules_per_row):
              col1, col2, col3 = st.columns(3)

              for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                  col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                  with col:
                      st.markdown(f"<h3 style='font-size: 13px;'>Ligand Name: {row['FDA drugnames']}</h3>", unsafe_allow_html=True)
                      st.image(row['Structures'], use_column_width=False, width=210)
                      st.write("MolLogP:", row['MolLogP'])
                      st.write("MolWt:", row['MolWt'])
                      st.write("NumRotatableBonds:", row['NumRotatableBonds'])
                      st.write("HBondDonors:", row['HBondDonors'])
                      st.write("HBondAcceptors:", row['HBondAcceptors'])
                      st.write("TPSA:", row['TPSA'])
                      st.write("HeavyAtoms:", row['HeavyAtoms'])
                      st.write("NumAromaticRings:", row['NumAromaticRings'])
                      st.write("SASA:", row['SASA'])
                      st.write("QED:", row['QED'])
                      st.write("Toxicity:", round(row['Tox-score'],3))
                      st.write("SA Score:", round(row['SAscore'],3))
                      st.write("---") 
      except:
        st.write("Dataframe uploaded is not in the proper format")     
    else:

        file_options = ["None","FDA Human", "Factor Xa","p38 alpha map kinase","human sirtuin 3","human sirtuin 2","human sirtuin 1"]
        selected_file = st.selectbox("Select CSV file", file_options)
    
    # Map selected file to the actual file name
        file_mapping = {
             "None":None,
             "FDA Human": "FDA_Human_2022-11-14_2.csv",
             "Factor Xa": "clean_mod/p_2p16_results_tdf_output_lipinski_clean.csv",
             "p38 alpha map kinase": "clean_mod/p_2rg5_pocket.pdb_results_output_lipinski_clean.csv",
             "human sirtuin 3": "clean_mod/p_4jsr.pdb_results_output_lipinski_clean.csv",
             "human sirtuin 2" :"clean_mod/p_5y0z_pocket.pdb_results_output_lipinski_clean.csv",
             "human sirtuin 1" :"clean_mod/p_4zzi_1ns_tdf_output_lipinski_clean.csv"
               }

       # Get the selected file name
        file_name = file_mapping[selected_file]  
            # Read the selected CSV file
        if file_name is None:
            st.write("Nothing is selected")
        else:
           df = pd.read_csv(file_name)       
           N = 12
           molecules_per_row = 3 
           session_state = SessionState.get(page_number = 0)
           last_page = len(df) // N
           prev, _ ,next = st.columns([1, 10, 1])
           if next.button("Next"):
               if session_state.page_number + 1 > last_page:
                   session_state.page_number = 0
               else:
                   session_state.page_number += 1
           if prev.button("Previous"):
               if session_state.page_number - 1 < 0:
                   session_state.page_number = last_page
               else:
                   session_state.page_number -= 1
           start = session_state.page_number * N 
           end = (1 + session_state.page_number) * N
           if session_state.page_number==0:
               df_graphs=generate_copy(df['Ligand SMILES'])
               non_empty_figures = []
               properties = ["MolLogP", "MolWt", "NumRotatableBonds", "HBondDonors", "HBondAcceptors",
                         "TPSA", "HeavyAtoms", "NumAromaticRings", "QED"]

               fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
               axes = axes.flatten()

               for i, prop in enumerate(properties):
                  ax = axes[i]
                  _, _, patches = ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                  ax.hist(df_graphs[prop], bins=13, alpha=0.7)
                  ax.set_title(prop, fontsize=25)
                  ax.set_xlabel("Value",fontsize=22 )
                  ax.set_ylabel("Frequency", fontsize=22)
                  ax.tick_params(axis='x', labelsize=18)
                  ax.tick_params(axis='y', labelsize=18)
               # Check if the figure is empty
                  if any(patches):
                     non_empty_figures.append(ax)
# Check if there are non-empty figures
               if non_empty_figures:
    # Display the non-empty figures
                  fig.subplots_adjust(hspace=0.5)
                  st.pyplot(fig)
               else:
                   pass
   
       # Slice the DataFrame based on the current page
           sliced_df = df[start:end].reset_index(drop=True)

        # Display the sliced DataFrame
           st.subheader("Original DataFrame Head")
           st.dataframe(sliced_df)

        # Process the DataFrame
           final_df = ret_final_df(sliced_df, model_file)

        # Display the structures with properties for the current page
        # Display the structures with properties for the current page
           for i in range(0, len(final_df), molecules_per_row):
               col1, col2, col3 = st.columns(3)

               for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                   col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                   with col:
                       st.markdown(f"<h3 style='font-size: 13px;'>Ligand Name: {row['FDA drugnames']}</h3>", unsafe_allow_html=True)
                       st.image(row['Structures'], use_column_width=False, width=210)
                       st.write("MolLogP:", row['MolLogP'])
                       st.write("MolWt:", row['MolWt'])
                       st.write("NumRotatableBonds:", row['NumRotatableBonds'])
                       st.write("HBondDonors:", row['HBondDonors'])
                       st.write("HBondAcceptors:", row['HBondAcceptors'])
                       st.write("TPSA:", row['TPSA'])
                       st.write("HeavyAtoms:", row['HeavyAtoms'])
                       st.write("NumAromaticRings:", row['NumAromaticRings'])
                       st.write("SASA:", row['SASA'])
                       st.write("QED:", row['QED'])
                       st.write("Toxicity:", round(row['Tox-score'],3))
                       st.write("SA Score:", round(row['SAscore'],3))
                       st.write("---")

def page_adme():

    st.title("ADME App - Knowdis")

    # Input field for molecule
    molecule_input = st.text_input("Enter a SMILES string:")
    mol=Chem.MolFromSmiles(molecule_input)

    if molecule_input:
        if mol is None:
            st.write("Invalid molecule.")
        else:
            temp_df=pd.DataFrame([molecule_input],columns=['smiles']).reset_index(drop=True)
            out_df=create_adme_dataframe_single(temp_df)
            st.markdown(
                         f"""
                         <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                           <p><strong>Absorption:</strong></p>
                           <p>Caco Permeability: {out_df['caco'][0]}</p>
                           <p>HIA Hou: {out_df['hia_hou'][0]}</p>
                           <p>PAMPA Permeability: {out_df['pampa'][0]}</p>
                           <p>Pgp Inhibition: {out_df['pgp'][0]}</p>
                           <p>Solubility: {out_df['solubility'][0]}</p>
                           <p>Lipophilicity: {out_df['lipo'][0]}</p>
                         </div>
                         <div>
                         <div style="background-color: #d5e8f7; padding: 5px; border-radius: 5px;">
                           <p><strong>Distribution:</strong></p>
                           <p>BBB Clearance: {out_df['bbb'][0]}</p>
                           <p>VDss : {out_df['vdss'][0]}</p>
                           <p>PPBR : {out_df['ppbr'][0]}</p>
                         </div>
                         <div>
                         <div style="background-color: #dbe5f1; padding: 5px; border-radius: 5px;">
                           <p><strong>Metabolism:</strong></p>
                           <p>CYP1A2 Inhibition: {out_df['CYP1A2_in'][0]}</p>
                           <p>CYP2C19 Inhibition: {out_df['CYP2C19_in'][0]}</p>
                           <p>CYP2D6 Substrate: {out_df['CYP2D6_sub'][0]}</p>
                           <p>CYP2D6 Inhibition: {out_df['CYP2D6_in'][0]}</p>
                           <p>CYP3A4 Substrate: {out_df['CYP3A4_sub'][0]}</p>
                           <p>CYP3A4 Inhibition: {out_df['CYP3A4_in'][0]}</p>
                         </div>
                         <div> 
                         <div style="background-color: #c9ddf4; padding: 5px; border-radius: 5px;">  
                           <p><strong>Excretion:</strong></p>
                           <p>Hepatocyte Clearance: {out_df['clear_hep'][0]}</p>
                           <p>Half-Life(Obach etal): {out_df['hl'][0]}</p>
                         </div>
                         <hr>
                         """,
                         unsafe_allow_html=True
                        ) 

                        # Display 2D structure
            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol, size=(300, 300)), use_column_width=False, width=300)


    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")
    st.write("Results will keep only the valid SMILES strings from the uploaded dataframe. Default dataframes can be selected for display if no file is uploaded.")                   
    file_options = ["None","FDA Human", "Factor Xa","p38 alpha map kinase","human sirtuin 3","human sirtuin 2","human sirtuin 1"]
    selected_file = st.selectbox("Select CSV file", file_options)
    if uploaded_file is not None:
        try:
            df=filter_valid_smiles(uploaded_file)
            N = 12
            session_state = SessionState.get(page_number = 0)
            molecules_per_row = 3 
            last_page = len(df) // N
            prev, _ ,next = st.columns([1, 10, 1])
            if next.button("Next"):
                if session_state.page_number + 1 > last_page:
                    session_state.page_number = 0
                else:
                    session_state.page_number += 1
            if prev.button("Previous"):
                if session_state.page_number - 1 < 0:
                    session_state.page_number = last_page
                else:
                    session_state.page_number -= 1
            start = session_state.page_number * N 
            end = (1 + session_state.page_number) * N

            final_df = create_adme_dataframe(df[start:end].reset_index(drop=True)) #Slice dataframe

                # Display the structures with properties for the current page
            for i in range(0, len(final_df), molecules_per_row):
                col1, col2, col3 = st.columns(3)

                for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                   col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                   with col:
            # Apply background color and padding to the column
                     st.image(row['Structures'], use_column_width=True, width=150)
                     col.markdown(
                         f"""
                         <h3 style="font-size: 13px;">Ligand Name: {row['FDA_Drug_Names']}</h3>
                         <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                           <p><strong>Absorption:</strong></p>
                           <p>Caco Permeability: {row['caco']}</p>
                           <p>HIA Hou: {row['hia_hou']}</p>
                           <p>PAMPA Permeability: {row['pampa']}</p>
                           <p>Pgp Inhibition: {row['pgp']}</p>
                           <p>Solubility: {row['solubility']}</p>
                           <p>Lipophilicity: {row['lipo']}</p>
                        </div>
                        <div>
                        <div style="background-color: #d5e8f7; padding: 5px; border-radius: 5px;">
                           <p><strong>Distribution:</strong></p>
                           <p>BBB Clearance: {row['bbb']}</p>
                           <p>VDss : {row['vdss']}</p>
                           <p>PPBR : {row['ppbr']}</p>
                        </div>
                        <div>
                        <div style="background-color: #dbe5f1; padding: 5px; border-radius: 5px;">
                           <p><strong>Metabolism:</strong></p>
                           <p>CYP1A2 Inhibition: {row['CYP1A2_in']}</p>
                           <p>CYP2C19 Inhibition: {row['CYP2C19_in']}</p>
                           <p>CYP2D6 Substrate: {row['CYP2D6_sub']}</p>
                           <p>CYP2D6 Inhibition: {row['CYP2D6_in']}</p>
                           <p>CYP3A4 Substrate: {row['CYP3A4_sub']}</p>
                           <p>CYP3A4 Inhibition: {row['CYP3A4_in']}</p>
                        </div>
                        <div> 
                        <div style="background-color: #c9ddf4; padding: 5px; border-radius: 5px;">  
                           <p><strong>Excretion:</strong></p>
                           <p>Hepatocyte Clearance: {row['clear_hep']}</p>
                           <p>Half-Life(Obach etal): {row['hl']}</p>
                        </div>
                        <hr>
                        """,
                        unsafe_allow_html=True
                     ) 
        except:
              st.write("Dataframe uploaded is not in the proper format")     
    else:
        file_mapping = {
            "None":None,
            "FDA Human": "FDA_Human_2022-11-14_2.csv",
            "Factor Xa": "clean_mod/p_2p16_results_tdf_output_lipinski_clean.csv",
            "p38 alpha map kinase": "clean_mod/p_2rg5_pocket.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 3": "clean_mod/p_4jsr.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 2" :"clean_mod/p_5y0z_pocket.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 1" :"clean_mod/p_4zzi_1ns_tdf_output_lipinski_clean.csv"
            }
         
            # Read the selected CSV file
        file_name = file_mapping[selected_file]

        if file_name is None:
            st.write("Nothing is selected")
            
        else: 
            df=filter_valid_smiles(file_name)    
            N = 12
            molecules_per_row = 3 
            session_state = SessionState.get(page_number = 0)
            last_page = len(df) // N
            prev, _ ,next = st.columns([1, 10, 1])
            if next.button("Next"):
                if session_state.page_number + 1 > last_page:
                    session_state.page_number = 0
                else:
                    session_state.page_number += 1
            if prev.button("Previous"):
                if session_state.page_number - 1 < 0:
                    session_state.page_number = last_page
                else:
                    session_state.page_number -= 1
            start = session_state.page_number * N 
            end = (1 + session_state.page_number) * N 

            final_df = create_adme_dataframe(df[start:end].reset_index(drop=True))

            # Display the structures with properties for the current page
            for i in range(0, len(final_df), molecules_per_row):
                col1, col2, col3 = st.columns(3)

                for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                    col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                    with col:
            # Apply background color and padding to the column
                      st.image(row['Structures'], use_column_width=True, width=150)
                      col.markdown(
                          f"""
                          <h3 style="font-size: 13px;">Ligand Name: {row['FDA_Drug_Names']}</h3>
                          <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                            <p><strong>Absorption:</strong></p>
                            <p>Caco Permeability: {row['caco']}</p>
                            <p>HIA Hou: {row['hia_hou']}</p>
                            <p>PAMPA Permeability: {row['pampa']}</p>
                            <p>Pgp Inhibition : {row['pgp']}</p>
                            <p>Solubility: {row['solubility']}</p>
                            <p>Lipophilicity: {row['lipo']}</p>
                          </div>
                          <div>
                          <div style="background-color: #d5e8f7; padding: 5px; border-radius: 5px;">
                            <p><strong>Distribution:</strong></p>
                            <p>BBB Clearance: {row['bbb']}</p>
                            <p>VDss : {row['vdss']}</p>
                            <p>PPBR : {row['ppbr']}</p>
                          </div>
                          <div>
                          <div style="background-color: #dbe5f1; padding: 5px; border-radius: 5px;">
                            <p><strong>Metabolism:</strong></p>
                            <p>CYP1A2 Inhibition: {row['CYP1A2_in']}</p>
                            <p>CYP2C19 Inhibition: {row['CYP2C19_in']}</p>
                            <p>CYP2D6 Substrate: {row['CYP2D6_sub']}</p>
                            <p>CYP2D6 Inhibition: {row['CYP2D6_in']}</p>
                            <p>CYP3A4 Substrate: {row['CYP3A4_sub']}</p>
                            <p>CYP3A4 Inhibition: {row['CYP3A4_in']}</p>
                          </div>
                          <div> 
                          <div style="background-color: #c9ddf4; padding: 5px; border-radius: 5px;">  
                            <p><strong>Excretion:</strong></p>
                            <p>Hepatocyte Clearance: {row['clear_hep']}</p>
                            <p>Half-Life(Obach etal): {row['hl']}</p>
                          </div>
                          <hr>
                          """,
                          unsafe_allow_html=True
                      )
                      
def create_tox21_dataframe(dframe):
    smiles = dframe['Ligand SMILES']
    new_df = pd.DataFrame({'smiles': smiles})
    new_df.to_csv("test_data.csv", index=False)
    checkpoint_dir='/home/beda/admet-master/toxicity/organ_toxicity/endocrine/checkpoints_features'
    preds_name='fnal.csv'
    run_chemprop_checkpoint(checkpoint_dir, preds_name)
    subprocess.run(['rm test_data.csv'],shell=True)
    df=pd.read_csv('fnal.csv')
    subprocess.run(['rm fnal.csv'],shell=True)
    return df


def app_tox():
    st.title("Toxicities App - Molecule AI")
     # Input field for molecule
    molecule_input = st.text_input("Enter a SMILES string:")
    mol=Chem.MolFromSmiles(molecule_input)

    if molecule_input:
        if mol is None:
            st.write("Invalid molecule.")
        else:
            temp_df=pd.DataFrame([molecule_input],columns=['smiles']).reset_index(drop=True)
            out_df=create_tox_dataframe_single(temp_df)
            st.markdown(
                         f"""
                         <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                           <p>Clintox Toxicity: {out_df['clintox'][0]}</p>
                           <p>Respiratory Toxicity: {out_df['respiratory'][0]}</p>
                           <p>Cardiotoxicity: {out_df['cardio'][0]}</p>
                           <p>Hepatotoxicity: {out_df['hepato'][0]}</p>
                           <p>Mutagenicity: {out_df['mutagen'][0]}</p>
                           <p>Carcinogenicity: {out_df['carci'][0]}</p>
                           <p>Mouse Oral LD50: {out_df['mouse_oral'][0]}</p>
                           <hr>
                         </div>
                         """,
                         unsafe_allow_html=True
                     ) 

            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol, size=(300, 300)), use_column_width=False, width=300)           

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")
    st.write("Results will keep only the valid SMILES strings from the uploaded dataframe. Default dataframes can be selected for display if no file is uploaded.")                   
    file_options = ["None","FDA Human", "Factor Xa","p38 alpha map kinase","human sirtuin 3","human sirtuin 2","human sirtuin 1"]
    selected_file = st.selectbox("Select CSV file", file_options)
    if uploaded_file is not None:
        try:
            df=filter_valid_smiles(uploaded_file)
            N = 12
            session_state = SessionState.get(page_number = 0)
            molecules_per_row = 3 
            last_page = len(df) // N
            prev, _ ,next = st.columns([1, 10, 1])
            if next.button("Next"):
                if session_state.page_number + 1 > last_page:
                    session_state.page_number = 0
                else:
                    session_state.page_number += 1
            if prev.button("Previous"):
                if session_state.page_number - 1 < 0:
                    session_state.page_number = last_page
                else:
                    session_state.page_number -= 1
            start = session_state.page_number * N 
            end = (1 + session_state.page_number) * N

            final_df = create_tox_dataframe(df[start:end].reset_index(drop=True)) #Slice dataframe

                # Display the structures with properties for the current page
            for i in range(0, len(final_df), molecules_per_row):
                col1, col2, col3 = st.columns(3)

                for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                   col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                   with col:
            # Apply background color and padding to the column
                     st.image(row['Structures'], use_column_width=True, width=150)
                     col.markdown(
                         f"""
                         <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                           <h3 style="font-size: 13px;">Ligand Name: {row['FDA_Drug_Names']}</h3>
                           <p>Clintox Toxicity: {row['clintox']}</p>
                           <p>Respiratory Toxicity: {row['respiratory']}</p>
                           <p>Cardiotoxicity: {row['cardio']}</p>
                           <p>Hepatotoxicity: {row['hepato']}</p>
                           <p>Mutagenicity: {row['mutagen']}</p>
                           <p>Carcinogenicity: {row['carci']}</p>
                           <p>Mouse Oral LD50: {row['mouse_oral']}</p>
                           <hr>
                         </div>
                         """,
                         unsafe_allow_html=True
                     )           
        except:
              st.write("Dataset is not in the proper format")   
    else:

        file_mapping = {
            "None":None,
            "FDA Human": "FDA_Human_2022-11-14_2.csv",
            "Factor Xa": "clean_mod/p_2p16_results_tdf_output_lipinski_clean.csv",
            "p38 alpha map kinase": "clean_mod/p_2rg5_pocket.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 3": "clean_mod/p_4jsr.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 2" :"clean_mod/p_5y0z_pocket.pdb_results_output_lipinski_clean.csv",
            "human sirtuin 1" :"clean_mod/p_4zzi_1ns_tdf_output_lipinski_clean.csv"
            }

        file_name = file_mapping[selected_file] 
            # Read the selected CSV file
        if file_name is None:
            st.write("Nothing is selected")
            
        else: 
            df=filter_valid_smiles(file_name)    
            N = 12
            molecules_per_row = 3 
            session_state = SessionState.get(page_number = 0)
            last_page = len(df) // N
            prev, _ ,next = st.columns([1, 10, 1])
            if next.button("Next"):
                if session_state.page_number + 1 > last_page:
                    session_state.page_number = 0
                else:
                    session_state.page_number += 1
            if prev.button("Previous"):
                if session_state.page_number - 1 < 0:
                    session_state.page_number = last_page
                else:
                    session_state.page_number -= 1
            start = session_state.page_number * N 
            end = (1 + session_state.page_number) * N 

            final_df = create_tox_dataframe(df[start:end].reset_index(drop=True))

                            # Display the structures with properties for the current page
            for i in range(0, len(final_df), molecules_per_row):
                col1, col2, col3 = st.columns(3)

                for j, (_, row) in enumerate(final_df.iloc[i:i + molecules_per_row].iterrows()):
                    col = col1 if j % 3 == 0 else col2 if j % 3 == 1 else col3
                    with col:
            # Apply background color and padding to the column
                      st.image(row['Structures'], use_column_width=True, width=150)
                      col.markdown(
                          f"""
                          <div style="flex=1; background-color: #b3e0f2; padding: 10px; border-radius: 10px;">
                            <h3 style="font-size: 13px;">Ligand Name: {row['FDA_Drug_Names']}</h3>
                            <p>Clintox Toxicity: {row['clintox']}</p>
                            <p>Respiratory Toxicity: {row['respiratory']}</p>
                            <p>Cardiotoxicity: {row['cardio']}</p>
                            <p>Hepatotoxicity: {row['hepato']}</p>
                            <p>Mutagenicity: {row['mutagen']}</p>
                            <p>Carcinogenicity: {row['carci']}</p>
                            <p>Mouse Oral LD50: {row['mouse_oral']}</p>
                            <hr>
                          </div>
                          """,
                          unsafe_allow_html=True
                      )

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def full_ana():
    st.title("Full ADMET with Tox 21 checkpoints")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
           df = pd.read_csv(uploaded_file)

        # Checkbox for filtering
           filtering = st.checkbox("Perform Filtering")

           if filtering:
               n_passes_frac = st.text_input("Enter fraction of filters to be passed(must be between 0 and 1):")
               total_f = 40
               if n_passes_frac:
                  tbp = math.floor(float(n_passes_frac) * total_f)    
                  if st.button("Run Analysis"):
                      st.text("Running analysis... Please wait.")
                      adme_df = create_adme_dataframe_ana(df)
                      tox_df = create_tox_dataframe_ana(df)
                      tox21 = create_tox21_dataframe(df)
                      tox21 = tox21.drop(['smiles'], axis=1)
                      semi_f = pd.concat([df, adme_df], axis=1)
                      final_df1 = pd.concat([semi_f, tox_df], axis=1)
                      final_df2 = pd.concat([final_df1, tox21], axis=1)
                      filtered_df = filter_dataframe(final_df2, tbp)
                      csv = convert_df(filtered_df)
                      st.download_button(
                          "Press to Download",
                          csv,
                          "output.csv",
                          "text/csv",
                          key='download-csv'
                      )

           else:
               if st.button("Run Analysis"):
                   st.text("Running analysis... Please wait.")
                   adme_df = create_adme_dataframe_ana(df)
                   tox_df = create_tox_dataframe_ana(df)
                   tox21 = create_tox21_dataframe(df)
                   tox21 = tox21.drop(['smiles'], axis=1)
                   semi_f = pd.concat([df, adme_df], axis=1)
                   final_df1 = pd.concat([semi_f, tox_df], axis=1)
                   final_df2 = pd.concat([final_df1, tox21], axis=1)
                   csv = convert_df(final_df2)
                   st.download_button(
                       "Press to Download",
                       csv,
                       "output.csv",
                       "text/csv",
                       key='download-csv'
                   )
        except:
            st.write("Uploaded dataframe is not in the proper format")


# Main app logic
def main():
    st.title("Molecule AI - ADME, Toxicity and Molecular Properties Suites")

    # Create navigation options
    page_selection = st.selectbox("Select an app page:", ("Main Page", "ADME App", "Toxicity App", "Molecular Properties App","Full Analysis(With Tox21)"))

    # Show the selected app page
    if page_selection == "Main Page":
        st.write("Welcome to the Main Page!")
    elif page_selection == "ADME App":
        page_adme()
    elif page_selection == "Toxicity App":
        app_tox()
    elif page_selection == "Molecular Properties App":
        molecular_properties()   
    elif page_selection =="Full Analysis(With Tox21)":
        full_ana()   

# Run the app
if __name__ == "__main__":
    main()

