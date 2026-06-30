import pandas as pd
import numpy as np
import joblib
import csv
from preprocessing_car import scale_new_data
import seaborn as sn
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import seaborn as sns
# Import the 'canonical' function from the module
from canonical_function import canonical
import matplotlib.pyplot as plt
#importing the cheminformatics module needed
import molvs
# Imbalanced-learn classifiers and ensembles
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
import base64
# XGBoost classifier
from xgboost import XGBClassifier

# CatBoost classifier
from catboost import CatBoostClassifier

# Scikit-learn ensemble
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
from preprocessing_mito import scale_new_data
from molvs import Standardizer, normalize
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import rdkit    
from Mold2_pywrapper import Mold2
import streamlit as st
import pandas as pd
import joblib
from preprocessing_car import scale_new_data
from preprocessing_dili import scale_new_data

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as Standardizer
from rdkit.Chem.SaltRemover import SaltRemover
from Standardizer import MyStandardizer
from molvs import Standardizer as MolVSStandardizer
from rdkit import Chem
import numpy as np
import pandas as pd

import ctxpy as ctx




# initialize once (IMPORTANT)
chem_client = ctx.Chemical(x_api_key="648a3d70")


#chem_client= ctx.Chemical(x_api_key='dd462a42-d747-464c-831d-d1c1dc8f14a4')


def get_toxprints(smiles):

    # safety check
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(729, dtype=np.uint8)

    try:
        result = ctx.search_toxprints(chemical=smiles)

        # DEBUG (remove later)
        print(f"ToxPrint returned type: {type(result)} for {smiles}")

        # CASE 1: DataFrame (your actual case)
        if isinstance(result, pd.DataFrame):

            fp = result.iloc[0].values.astype(np.uint8)

            # DEBUG
            print(f"ON bits: {np.sum(fp)}")

            return fp

        # CASE 2: numpy / list fallback
        fp = np.array(result).flatten().astype(np.uint8)

        print(f"ON bits (fallback): {np.sum(fp)}")

        return fp

    except Exception as e:

        print(f"❌ ToxPrint failed for {smiles}: {e}")

        return np.zeros(729, dtype=np.uint8)


def compute_toxprints(smiles_list):

    fps = []

    for i, smi in enumerate(smiles_list):

        print(f"Generating ToxPrints for [{i+1}/{len(smiles_list)}]: {smi}")

        fp = get_toxprints(smi)

        print(f"Final ON bits: {np.sum(fp)}")

        fps.append(fp)

    return fps







def jaccard_similarity(fp1, fp2):

    intersection = np.sum(
        np.logical_and(fp1, fp2)
    )

    union = np.sum(
        np.logical_or(fp1, fp2)
    )

    if union == 0:
        return 0.0

    return intersection / union 


calibrated_car_model = joblib.load(
    "calibrated_eec_catboost_model.pkl"
)

calibrated_dili_model = joblib.load(
    "calibrated_dili_eec_xgb_model.pkl"
)

coral = joblib.load(
    "coral_car_aligner.pkl"
)


fp_cal_clean = joblib.load(
    "toxprint_calibration_fps.pkl"
)

y_cal_clean = joblib.load(
    "toxprint_calibration_labels.pkl"
)

cal_nc = joblib.load(
    "toxprint_calibration_nc.pkl"
)

weights = joblib.load(
    "ensemble_weights.pkl"
)

w_car = weights["w_car"]
w_dili = weights["w_dili"]

calibrated_car_model = joblib.load(
    "calibrated_eec_catboost_model.pkl"
)

calibrated_dili_model = joblib.load(
    "calibrated_dili_eec_xgb_model.pkl"
)








# === Heavy Metals List ===
heavy_metals_atomic_nums = [
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88,
    64, 33, 50
]

# === Preprocessing Functions ===

# def remove_heavy_metals(df):
#     def has_no_heavy_metals(smiles):
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False
#         return not any(atom.GetAtomicNum() in heavy_metals_atomic_nums for atom in mol.GetAtoms())
#     return df[df['smiles'].apply(has_no_heavy_metals)]

def remove_heavy_metals(df):

    def classify(smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "invalid_smiles"

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in heavy_metals_atomic_nums:
                return "heavy_metal"

        return "ok"

    df = df.copy()
    df["qc_flag"] = df["smiles"].apply(classify)

    return df

metal_disconnector = MolVSStandardizer().disconnect_metals

def disconnect_metals_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = metal_disconnector(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

# === Solvent Definitions ===
solvents = {
    'Water': 'O', 'Methanol': 'CO', 'Ethanol': 'CCO', 'Isopropanol': 'CC(O)C',
    'Acetic Acid': 'CC(=O)O', 'Formic Acid': 'C(=O)O', 'DMSO': 'CS(=O)C', 'Acetonitrile': 'CC#N',
    'DMF': 'CN(C)C=O', 'Acetone': 'CC(=O)C', 'THF': 'C1CCOCC1', 'Ethyl Acetate': 'CCOC(=O)C',
    'MEK': 'CCC(=O)C', 'Hexane': 'CCCCCC', 'Toluene': 'CC1=CC=CC=C1', 'Diethyl Ether': 'CCOCC',
    'Chloroform': 'ClC(Cl)Cl', 'Benzene': 'C1=CC=CC=C1', 'Cyclohexane': 'C1CCCCC1',
    'Dichloromethane': 'ClCCl', 'Carbon Tetrachloride': 'ClC(Cl)(Cl)Cl', '1,2-Dichloroethane': 'ClCCCl'
}

solvent_mols = {name: Chem.MolFromSmiles(smiles) for name, smiles in solvents.items()}

def remove_solvents(compound_smiles, solvent_mols):
    mol = Chem.MolFromSmiles(compound_smiles)
    if not mol:
        return None

    for solvent_mol in solvent_mols.values():
        if Chem.MolToSmiles(mol) == Chem.MolToSmiles(solvent_mol):
            return compound_smiles

    fragments = Chem.GetMolFrags(mol, asMols=True)
    non_solvent_fragments = []
    for fragment in fragments:
        is_solvent = any(Chem.MolToSmiles(fragment) == Chem.MolToSmiles(solv) for solv in solvent_mols.values())
        if not is_solvent:
            non_solvent_fragments.append(fragment)

    if not non_solvent_fragments:
        return None

    # Corrected: Combine all fragments without duplicating
    combined = non_solvent_fragments[0]
    for frag in non_solvent_fragments[1:]:
        combined = Chem.CombineMols(combined, frag)

    try:
        Chem.SanitizeMol(combined)
    except Exception:
        return None

    return Chem.MolToSmiles(combined)




# === Salt Removal and Sanitization ===
def remove_salts_and_sanitize(smiles):
    salt_remover = SaltRemover()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    cleaned_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
        return smiles
    try:
        Chem.SanitizeMol(cleaned_mol)
    except Exception:
        return smiles
    return Chem.MolToSmiles(cleaned_mol)


# === Helper function to validate carbon count and molecular weight ===
from rdkit.Chem import Descriptors

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # Invalid SMILES

    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    if num_carbons < 4:
        return False

    if Descriptors.MolWt(mol) > 909:
        return False

    return True





# Initialize custom and MolVS standardizers
Standardizer = MyStandardizer()
standardizer = MolVSStandardizer()

# Define the final preprocessing function
def preprocess_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharged_mol = standardizer.uncharge(mol)
    canonical_tautomer_mol = standardizer.canonicalize_tautomer(uncharged_mol)
    return Chem.MolToSmiles(canonical_tautomer_mol)


# Define the columns to convert after descriptor generation
columns_to_convert = ['D018', 'D326', 'D330', 'D340']

def convert_columns_to_int64(df):
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col].astype('int64')

# Define the columns to convert after descriptor generation
columns_to_convert = ['D018', 'D326', 'D330', 'D340']


def check_disconnected_smiles(smiles):
    print(f"Checking SMILES: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fragments = Chem.GetMolFrags(mol, asMols=False)
        return len(fragments) > 1
    else:
        print(f"Invalid SMILES: {smiles}")
        return False


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    
    if mol is not None:
        # Convert the sanitized molecule to canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return canonical_smiles
    else:
        print(f"Unable to parse SMILES: {smiles}")
        # Log problematic SMILES
        return None
    
    
def generate_mold2_descriptors(smiles_series):
    # Initialize Mold2
    mold2 = Mold2()

    # Convert SMILES to Mol objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_series]

    # Calculate descriptors
    descriptors = mold2.calculate(mols)

    # Convert the result into a DataFrame if needed
    descriptors_df = pd.DataFrame(descriptors)

    return descriptors_df


# Load preprocessing objects
@st.cache_resource
def load_preprocessing_objects_car():
    selector = joblib.load("selector_car.pkl")
    scaler = joblib.load("scaler_car.pkl")
    float_cols = joblib.load("float_cols_car.pkl")
    retained_features_final = joblib.load("retained_features_final_car.pkl")
    return selector, scaler, float_cols, retained_features_final

# Define preprocessing function for new data
def preprocess_new_data_car(raw_input_df):
    selector, scaler, float_cols, retained_features_final = load_preprocessing_objects_car()

    # Step 1: Apply variance threshold selection
    retained_columns = raw_input_df.columns[selector.get_support(indices=True)]
    var_filtered_df = raw_input_df[retained_columns]

    # Step 2: Scale float features
    scaled_df = scale_new_data(var_filtered_df, scaler, float_cols)

    # Step 3: Apply correlation-based feature selection
    final_df = scaled_df[retained_features_final]

    return final_df


# Load preprocessing objects
@st.cache_resource
def load_preprocessing_objects_dili():
    selector = joblib.load("selector_dili.pkl")
    scaler = joblib.load("scaler_dili.pkl")
    float_cols = joblib.load("float_cols_dili.pkl")
    retained_features_final = joblib.load("retained_features_final_dili.pkl")
    return selector, scaler, float_cols, retained_features_final

# Define preprocessing function for new data
def preprocess_new_data_dili(raw_input_df):
    selector, scaler, float_cols, retained_features_final = load_preprocessing_objects_dili()

    # Step 1: Apply variance threshold selection
    retained_columns = raw_input_df.columns[selector.get_support(indices=True)]
    var_filtered_df = raw_input_df[retained_columns]

    # Step 2: Scale float features
    scaled_df = scale_new_data(var_filtered_df, scaler, float_cols)

    # Step 3: Apply correlation-based feature selection
    final_df = scaled_df[retained_features_final]

    return final_df


# Custom list of important features to retain after preprocessing

important_features_car = ['D001', 'D004', 'D005', 'D013', 'D014', 'D016', 'D018', 'D019', 'D025', 'D026', 'D027', 
                          'D034', 'D035', 'D053', 'D122', 'D123', 'D128', 'D130', 'D134', 'D154', 'D164', 'D173', 
                          'D176', 'D186', 'D187', 'D194', 'D195', 'D197', 'D199', 'D252', 'D259', 'D268', 'D274', 
                          'D279', 'D282', 'D308', 'D310', 'D322', 'D323', 'D325', 'D326', 'D327', 'D330', 'D335', 
                          'D336', 'D337', 'D338', 'D339', 'D360', 'D366', 'D367', 'D368', 'D369', 'D370', 'D371', 
                          'D372', 'D373', 'D374', 'D375', 'D376', 'D377', 'D378', 'D379', 'D380', 'D381', 'D382', 
                          'D383', 'D384', 'D385', 'D389', 'D392', 'D393', 'D394', 'D395', 'D396', 'D397', 'D399', 
                          'D411', 'D413', 'D447', 'D450', 'D451', 'D452', 'D453', 'D454', 'D459', 'D460', 'D461', 
                          'D462', 'D468', 'D469', 'D470', 'D475', 'D476', 'D477', 'D478', 'D480', 'D483', 'D484', 
                          'D485', 'D486', 'D492', 'D493', 'D494', 'D499', 'D500', 'D501', 'D502', 'D503', 'D504', 
                          'D505', 'D506', 'D507', 'D508', 'D509', 'D510', 'D511', 'D532', 'D533', 'D534', 'D535', 
                          'D536', 'D537', 'D538', 'D539', 'D540', 'D541', 'D542', 'D544', 'D545', 'D546', 'D547', 
                          'D550', 'D551', 'D556', 'D559', 'D560', 'D561', 'D562', 'D570', 'D573', 'D574', 'D575', 
                          'D576', 'D578', 'D579', 'D582', 'D588', 'D589', 'D590', 'D591', 'D596', 'D597', 'D598', 
                          'D599', 'D600', 'D601', 'D602', 'D604', 'D605', 'D606', 'D607', 'D621', 'D648', 'D649', 
                          'D650', 'D651', 'D652', 'D668', 'D674', 'D675', 'D677', 'D679', 'D680', 'D689', 'D708', 
                          'D709', 'D712', 'D713', 'D714', 'D715', 'D716', 'D717', 'D718', 'D719', 'D721', 'D724', 
                          'D729', 'D730', 'D731', 'D732', 'D733', 'D738', 'D739', 'D744', 'D745', 'D746', 
                          'D748', 'D749', 'D752', 'D754', 'D756', 'D763', 'D765', 'D768', 'D774', 'D775', 'D777']

        




@st.cache_resource
def load_coral():
    return joblib.load("coral_car_aligner.pkl")

coral = load_coral()

cytotoxic_car = ['AD013', 'AD034', 'AD053', 'AD123', 'AD308', 'AD330', 'AD360', 'AD366', 'AD367', 
                 'AD368', 'AD372', 'AD373', 'AD375', 'AD377', 'AD379', 'AD380', 'AD381', 'AD382', 
                 'AD385', 'AD389', 'AD392', 'AD396', 'AD397', 'AD483', 'AD533', 'AD534', 'AD535', 
                 'AD536', 'AD537', 'AD538', 'AD539', 'AD541', 'AD542', 'AD544', 'AD556', 'AD570', 
                 'AD573', 'AD574', 'AD575', 'AD605', 'AD648', 'AD668', 'AD679', 'AD689', 'AD712', 
                 'AD713', 'AD715', 'AD746', 'AD768', 'AD774', 'AD775', 'D001', 'D004', 'D005', 'D013', 
                 'D014', 'D016', 'D018', 'D019', 'D025', 'D026', 'D035', 'D123', 'D128', 'D130', 'D134', 
                 'D164', 'D173', 'D176', 'D186', 'D187', 'D194', 'D195', 'D197', 'D199', 'D252', 'D259', 
                 'D268', 'D274', 'D279', 'D282', 'D308', 'D310', 'D322', 'D323', 'D325', 'D326', 'D327', 
                 'D335', 'D336', 'D337', 'D338', 'D339', 'D360', 'D366', 'D367', 'D368', 'D369', 'D370', 
                 'D371', 'D372', 'D373', 'D374', 'D375', 'D377', 'D378', 'D380', 'D381', 'D384', 'D385', 
                 'D393', 'D395', 'D396', 'D413', 'D447', 'D450', 'D451', 'D452', 'D453', 'D454', 'D459', 
                 'D460', 'D461', 'D462', 'D468', 'D469', 'D470', 'D475', 'D476', 'D477', 'D478', 'D480', 
                 'D483', 'D484', 'D485', 'D486', 'D492', 'D493', 'D494', 'D499', 'D500', 'D501', 'D502', 
                 'D503', 'D504', 'D505', 'D506', 'D507', 'D508', 'D509', 'D510', 'D532', 'D533', 'D534', 
                 'D535', 'D536', 'D537', 'D538', 'D539', 'D544', 'D545', 'D546', 'D547', 'D550', 'D551', 
                 'D556', 'D559', 'D560', 'D561', 'D562', 'D570', 'D574', 'D575', 'D576', 'D578', 'D579', 
                 'D582', 'D588', 'D589', 'D590', 'D591', 'D596', 'D597', 'D598', 'D599', 'D600', 'D601', 
                 'D602', 'D604', 'D605', 'D606', 'D607', 'D621', 'D648', 'D649', 'D650', 'D651', 'D652', 
                 'D668', 'D674', 'D675', 'D677', 'D679', 'D680', 'D689', 'D708', 'D712', 'D714', 'D715', 
                 'D716', 'D717', 'D718', 'D719', 'D721', 'D724', 'D729', 'D730', 'D731', 'D732', 'D733', 'D738', 'D739', 'D744', 'D745', 
                 'D746', 'D748', 'D749', 'D754', 'D756', 'D763', 'D765', 'D768', 'D774', 'D777']

important_features_dili = ['D001', 'D004', 'D005', 'D013', 'D014', 'D016', 'D018', 'D019', 'D025', 'D026', 'D027', 'D034', 'D035', 'D123', 
                           'D128', 'D130', 'D164', 'D173', 'D176', 'D186', 'D194', 'D195', 'D197', 'D199', 'D237', 'D259', 'D268', 'D274', 
                           'D279', 'D282', 'D284', 'D287', 'D308', 'D322', 'D323', 'D324', 'D325', 'D326', 'D327', 'D328', 'D335', 'D336', 
                           'D337', 'D338', 'D339', 'D340', 'D360', 'D366', 'D367', 'D368', 'D369', 'D370', 'D371', 'D372', 'D373', 'D374', 
                           'D375', 'D376', 'D377', 'D378', 'D379', 'D380', 'D381', 'D383', 'D384', 'D385', 'D388', 'D389', 'D392', 'D393', 
                           'D396', 'D411', 'D413', 'D421', 'D454', 'D461', 'D462', 'D469', 'D470', 'D477', 'D478', 'D481', 'D482', 'D483', 
                           'D484', 'D485', 'D486', 'D494', 'D500', 'D501', 'D502', 'D503', 'D504', 'D505', 'D506', 'D507', 'D508', 'D509', 
                           'D510', 'D532', 'D533', 'D534', 'D535', 'D536', 'D537', 'D538', 'D539', 'D546', 'D547', 'D559', 'D560', 'D561', 
                           'D568', 'D569', 'D574', 'D575', 'D576', 'D579', 'D588', 'D589', 'D590', 'D591', 'D596', 'D597', 'D598', 'D599', 
                           'D600', 'D601', 'D602', 'D604', 'D606', 'D607', 'D619', 'D621', 'D625', 'D627', 'D643', 'D647', 'D649', 'D650', 
                           'D651', 'D674', 'D675', 'D676', 'D677', 'D678', 'D679', 'D680', 'D689', 'D708', 'D712', 'D714', 'D715', 'D716', 
                           'D717', 'D718', 'D719', 'D721', 'D722', 'D724', 'D729', 'D730', 'D731', 'D732', 'D733', 'D738', 'D739', 'D742', 
                           'D743', 'D744', 'D745', 'D746', 'D748', 'D749', 'D750', 'D753', 'D754', 'D756', 'D763', 'D765', 'D774', 'D777']



#function to generate csv file
def file_download(data, file):
    df = data.to_csv(index=False)
    f=base64.b64encode(df.encode()).decode()
    link= f'<a href = "data:file/csv; base64,{f}" download={file}> Download{file} file</a>'
    return link


st.set_page_config(page_title='Drug Induced Hepatotoxicity Prediction App', layout='wide')

# Styled sidebar header
st.sidebar.markdown(
    '<h2 style="color: white; background-color: #6A1B9A; padding: 12px; border-radius: 10px; text-align: center;">'
    'Use this Sidebar for Hepatotoxicity Prediction</h2>',
    unsafe_allow_html=True
)

# Developer credit section
st.markdown(
    """
    <div style="background-color:#F3E5F5; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <p style="color:#4A148C; font-size: 16px;">
            This Web Application was developed by 
            <a href="https://www.linkedin.com/in/emmanuel-ogbuewu-18a3a4117/" target="_blank" style="color:#1A237E; text-decoration: none;">
                Emmanuel I. Ogbuewu
            </a>, a PhD student of Dr. Jeremy S. Edwards at the University of New Mexico.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Description of the app’s purpose
st.markdown(
    """
    <div style="background-color:#E8EAF6; padding: 15px; border-radius: 10px;">
        <p style="color:#1A237E; font-size: 16px;">
            Hepatotoxicity, or drug-induced liver injury, is a major reason for drug withdrawals and clinical trial failures. 
            Predicting liver toxicity early is critical for ensuring patient safety and improving drug development success. 
            Our tool helps identify potential hepatotoxic risks of drug candidates using advanced machine learning. 
            See the chemical space visualization of the training compounds below.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)




def plot_umap_coral(
    car_umap,
    dili_umap,
    dili_align_umap,
    title="UMAP Visualization of CORAL"
):
    """
    Plot UMAP before and after CORAL alignment.
    Returns matplotlib figure.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before CORAL
    axes[0].scatter(
        car_umap["UMAP_1"],
        car_umap["UMAP_2"],
        c="blue",
        marker="x",
        label="Target (CAR Antagonist)",
        alpha=0.8
    )

    axes[0].scatter(
        dili_umap["UMAP_1"],
        dili_umap["UMAP_2"],
        c="red",
        marker="o",
        label="Source (DILI)",
        alpha=0.6
    )

    axes[0].set_title("Before CORAL")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend()

    # After CORAL
    axes[1].scatter(
        car_umap["UMAP_1"],
        car_umap["UMAP_2"],
        c="blue",
        marker="x",
        label="Target (CAR Antagonist)",
        alpha=0.8
    )

    axes[1].scatter(
        dili_align_umap["UMAP_1"],
        dili_align_umap["UMAP_2"],
        c="red",
        marker="o",
        label="Source (DILI)",
        alpha=0.6
    )

    axes[1].set_title("After CORAL")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

X_car_umap = pd.read_csv("X_car_umap.csv")
X_DILI_umap = pd.read_csv("X_DILI_umap.csv")
X_DILI_align_umap = pd.read_csv("X_DILI_align_umap.csv")


fig = plot_umap_coral(
    X_car_umap,
    X_DILI_umap,
    X_DILI_align_umap,
    title="CORrelation ALignment (CORAL) between CAR Antagonist & DILI Training Dataset"
)

st.pyplot(fig)










one_or_few_SMILES = st.sidebar.text_input('Enter single SMILE strings in single or double quotation separated by comma:', "['CCO']")
st.sidebar.markdown('''"or upload SMILE strings in CSV format, note that the SMILE strings of the molecules should be in 'smiles' column:"''')
many_SMILES = st.sidebar.file_uploader("================================")

st.sidebar.markdown("""**if you upload the csv file, click the button below to get the hepatotoxicity prediction**""")
predict_button = st.sidebar.button("Predict Drug Induced Hepatotoxicity")

# ============================================================
# MAIN LOGIC
# ============================================================

if one_or_few_SMILES != "['CCO']":

    try:

        smiles_list = ast.literal_eval(one_or_few_SMILES)
        df = pd.DataFrame(smiles_list, columns=["smiles"])

        # ====================================================
        # STEP 1: REMOVE HEAVY METALS
        # ====================================================



        df = remove_heavy_metals(df)

        clean = df[df["qc_flag"] == "ok"]
        invalid = df[df["qc_flag"] == "invalid_smiles"]
        metals = df[df["qc_flag"] == "heavy_metal"]

        if len(clean) == 0:
            if len(invalid) > 0:
                st.error("❌ Invalid SMILES format detected.")
            elif len(metals) > 0:
                st.error("❌ Heavy metal-containing molecules detected.")
            else:
                st.error("❌ No valid molecules after preprocessing.")

                st.stop()

            st.success("✅ Heavy metal filtering completed.")

        
        # df = remove_heavy_metals(df)

        # if df.empty:
        #     st.error("❌ All input SMILES contain heavy metals.")
        #     st.stop()

        # st.success("✅ Heavy metals filtered.")

        # ====================================================
        # STEP 2: DISCONNECT METALS
        # ====================================================

        df["smiles"] = df["smiles"].apply(disconnect_metals_from_smiles)

        st.success("✅ Metals disconnected.")

        # ====================================================
        # STEP 3: REMOVE SOLVENTS
        # ====================================================

        df["smiles"] = df["smiles"].apply(lambda s: remove_solvents(s, solvent_mols))

        df.dropna(subset=["smiles"], inplace=True)

        if df.empty:
            st.error("❌ All input SMILES removed after solvent filtering.")
            st.stop()

        st.success("✅ Solvents removed.")

        # ====================================================
        # STEP 4: REMOVE SALTS
        # ====================================================

        df["smiles"] = df["smiles"].apply(remove_salts_and_sanitize)

        df.dropna(subset=["smiles"], inplace=True)

        if df.empty:
            st.error("❌ No valid compounds remaining after salt removal.")
            st.stop()

        st.success("✅ Salts removed and SMILES sanitized.")

        # ====================================================
        # STEP 5: CHECK DISCONNECTED COMPONENTS
        # ====================================================

        disconnected = df["smiles"].apply(check_disconnected_smiles)

        if disconnected.any():
            st.error("❌ Some input SMILES contain disconnected components.")
            st.stop()

        st.success("✅ No disconnected components found.")

        # ====================================================
        # STEP 6: VALIDATE COMPOUNDS
        # ====================================================

        valid_mask = df["smiles"].apply(lambda s: s is not None and is_valid_smiles(s))

        if not valid_mask.all():

            invalid_smiles = df.loc[~valid_mask, "smiles"]

            for smi in invalid_smiles:

                mol = Chem.MolFromSmiles(smi)

                if mol is None:

                    st.error(f"❌ Invalid SMILES: {smi}")

                else:

                    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
                    mw = Descriptors.MolWt(mol)

                    if num_carbons < 4:
                        st.error(f"❌ Number of carbons in compound {smi} is less than 4.")

                    if mw > 909:
                        st.error(f"❌ Molecular weight of compound {smi} is greater than 909.")

            st.stop()

        st.success("✅ All compounds passed validation.")

        # ====================================================
        # STEP 7: STANDARDIZE SMILES
        # ====================================================

        df["smiles"] = Standardizer.standardize_smiles(df["smiles"].tolist())

        st.success("✅ Input SMILES standardized.")

        # ====================================================
        # STEP 8: PREPROCESS MOLECULES
        # ====================================================

        df["smiles"] = df["smiles"].apply(preprocess_molecule)

        df.dropna(subset=["smiles"], inplace=True)

        st.success("✅ Molecules preprocessed.")

        # ====================================================
        # STEP 9: CANONICALIZE
        # ====================================================

        df["smiles"] = df["smiles"].apply(canonical)

        st.success("✅ Canonical SMILES generated.")

        # ====================================================
        # STEP 10: GENERATE MOLD2 DESCRIPTORS
        # ====================================================

        descriptors_df = generate_mold2_descriptors(df["smiles"])

        convert_columns_to_int64(descriptors_df)

        st.success("✅ Mold2 descriptors generated.")

        # ====================================================
        # STEP 11: PREPROCESS FOR CAR + DILI
        # ====================================================

        processed_car_df = preprocess_new_data_car(descriptors_df)

        processed_dili_df = preprocess_new_data_dili(descriptors_df)

        st.success("✅ Preprocessing complete.")

        # ====================================================
        # STEP 12: IMPORTANT FEATURES
        # ====================================================

        selected_car_df = processed_car_df[[col for col in important_features_car if col in processed_car_df.columns]]

        selected_dili_df = processed_dili_df[[col for col in important_features_dili if col in processed_dili_df.columns]]

        st.success("✅ Important features selected.")
        
       

        # ====================================================
        # STEP 13: CORAL ALIGNMENT
        # ====================================================

        selected_car_df_align = coral.transform(selected_car_df)

        selected_car_df_align = pd.DataFrame(
            selected_car_df_align,
            columns=important_features_car,
            index=selected_car_df.index
        )

        selected_car_df_align.columns = ["A" + col for col in selected_car_df_align.columns]

        selected_car_df_full = pd.concat(
            [selected_car_df, selected_car_df_align],
            axis=1
        )

        st.success("✅ CORAL alignment complete.")

        # ====================================================
        # STEP 14: CAR CYTOTOXIC FEATURES
        # ====================================================

        missing_features = [col for col in cytotoxic_car if col not in selected_car_df_full.columns]

        if missing_features:

            st.error(f"❌ Missing required CAR features: {missing_features}")

            st.stop()

        selected_car_df_final = selected_car_df_full[cytotoxic_car]

        st.success("✅ CAR antagonist cytotoxic features selected.")

        

        # ====================================================
        # STEP 15: MODEL PROBABILITIES
        # ====================================================
        

        
        proba_car = calibrated_car_model.predict_proba(selected_car_df_final)

        proba_dili = calibrated_dili_model.predict_proba(selected_dili_df)

        st.success("✅ CAR and DILI probabilities generated.")

        # ====================================================
        # STEP 16: WEIGHTED SOFT VOTING
        # ====================================================

        ensemble_proba = (
            w_car * proba_car +
            w_dili * proba_dili
        )

        ensemble_prediction = np.argmax(
            ensemble_proba,
            axis=1
        )

        st.success("✅ Ensemble prediction complete.")

        # ====================================================
        # STEP 17: MCP SETUP
        # ====================================================

        alpha = 0.10

        idx0 = np.where(y_cal_clean == 0)[0]

        idx1 = np.where(y_cal_clean == 1)[0]

        k0 = max(1, int(np.sqrt(len(idx0))))

        k1 = max(1, int(np.sqrt(len(idx1))))

        test_fps = compute_toxprints(df["smiles"])

        p0_list = []

        p1_list = []

        set_labels = []

        # ====================================================
        # STEP 18: MCP LOOP
        # ====================================================

        for i, test_fp in enumerate(test_fps):

            sims = np.array([
                jaccard_similarity(test_fp, fp)
                for fp in fp_cal_clean
            ])

            # ------------------------------------------------
            # CLASS 0
            # ------------------------------------------------

            sims0 = sims[idx0]

            nn0_local = np.argsort(sims0)[-k0:]

            nn0 = idx0[nn0_local]

            weights0 = sims[nn0]

            if np.sum(weights0) == 0:

                p0 = 0.0

            else:

                weights0 = weights0 / np.sum(weights0)

                scores0 = cal_nc[nn0]

                nc0 = 1 - ensemble_proba[i, 0]

                p0 = np.sum(weights0[scores0 >= nc0])

            # ------------------------------------------------
            # CLASS 1
            # ------------------------------------------------

            sims1 = sims[idx1]

            nn1_local = np.argsort(sims1)[-k1:]

            nn1 = idx1[nn1_local]

            weights1 = sims[nn1]

            if np.sum(weights1) == 0:

                p1 = 0.0

            else:

                weights1 = weights1 / np.sum(weights1)

                scores1 = cal_nc[nn1]

                nc1 = 1 - ensemble_proba[i, 1]

                p1 = np.sum(weights1[scores1 >= nc1])

            pred_set = []

            if p0 > alpha:
                pred_set.append("0")

            if p1 > alpha:
                pred_set.append("1")

            if len(pred_set) == 0:

                set_labels.append("{}")

            else:

                set_labels.append("{" + ",".join(pred_set) + "}")

            p0_list.append(p0)

            p1_list.append(p1)

        # ====================================================
        # STEP 19: RESULTS TABLE
        # ====================================================

        results_df = pd.DataFrame({

            "SMILES": df["smiles"],

            "CAR_Probability": proba_car[:, 1],

            "DILI_Probability": proba_dili[:, 1],

            "Ensemble_Probability": ensemble_proba[:, 1],

            "Prediction": np.where(
                ensemble_prediction == 1,
                "DILI Positive",
                "DILI Negative"
            ),

            "MCP_Set": set_labels,

            "p_value_0": p0_list,

            "p_value_1": p1_list

        })

        # ====================================================
        # STEP 20: CONFIDENCE LABELS
        # ====================================================

        results_df["Confidence"] = np.select(

            [
                results_df["MCP_Set"] == "{1}",
                results_df["MCP_Set"] == "{0}",
                results_df["MCP_Set"] == "{0,1}",
                results_df["MCP_Set"] == "{}"
            ],

            [
                "Confident DILI",
                "Confident Non-DILI",
                "Ambiguous",
                "Evidence Failure"
            ],

            default="Unknown"

        )

        # ====================================================
        # STEP 21: DISPLAY SINGLE COMPOUND RESULT
        # ====================================================

        if len(results_df) == 1:

            st.subheader("DILI Prediction")

            st.write(f"Prediction: {results_df.iloc[0]['Prediction']}")

            st.write(f"Ensemble Probability: {results_df.iloc[0]['Ensemble_Probability']:.3f}")

            st.write(f"P-value Inactive: {results_df.iloc[0]['p_value_0']:.3f}")

            st.write(f"P-value Active: {results_df.iloc[0]['p_value_1']:.3f}")

            st.write(f"MCP Set: {results_df.iloc[0]['MCP_Set']}")

            st.write(f"Confidence Category: {results_df.iloc[0]['Confidence']}")

        # ====================================================
        # STEP 22: DISPLAY RESULTS
        # ====================================================

        st.dataframe(results_df)

        st.sidebar.markdown("## Prediction Results")

        st.sidebar.write(results_df)

        st.sidebar.markdown(
            file_download(
                results_df,
                "hepatotoxicity_prediction.csv"
            ),
            unsafe_allow_html=True
        )

        st.success("✅ Weighted Soft Voting + ToxPrint kNN-MCP prediction completed.")

    except Exception as e:

        st.error(f"⚠️ Failed to process input: {e}")
    
    
elif predict_button:

    if many_SMILES is not None:

        try:

            df2 = pd.read_csv(many_SMILES)

            if "smiles" not in df2.columns:
                st.error("❌ Uploaded CSV must contain a 'smiles' column.")
                st.stop()

            # ====================================================
            # STEP 1: REMOVE HEAVY METALS
            # ====================================================
            





            df2 = remove_heavy_metals(df2)

            clean = df[df2["qc_flag"] == "ok"]
            invalid = df2[df2["qc_flag"] == "invalid_smiles"]
            metals = df2[df2["qc_flag"] == "heavy_metal"]

            if len(clean) == 0:
                if len(invalid) > 0:
                    st.error("❌ Invalid SMILES format detected.")
                elif len(metals) > 0:
                    st.error("❌ Heavy metal-containing molecules detected.")
                else:
                    st.error("❌ No valid molecules after preprocessing.")
                    st.stop()
                st.success("✅ Heavy metal filtering completed.")
            
            # df2 = remove_heavy_metals(df2)
            # if df2.empty:
            #     st.error("❌ All uploaded SMILES contain heavy metals.")
            #     st.stop()
            # st.success("✅ Heavy metals filtered.")

            # ====================================================
            # STEP 2: DISCONNECT METALS
            # ====================================================

            df2["smiles"] = df2["smiles"].apply(disconnect_metals_from_smiles)
            st.success("✅ Metals disconnected.")

            # ====================================================
            # STEP 3: REMOVE SOLVENTS
            # ====================================================

            df2["smiles"] = df2["smiles"].apply(lambda s: remove_solvents(s, solvent_mols))
            df2.dropna(subset=["smiles"], inplace=True)

            if df2.empty:
                st.error("❌ All SMILES removed after solvent filtering.")
                st.stop()

            st.success("✅ Solvents removed.")

            # ====================================================
            # STEP 4: REMOVE SALTS
            # ====================================================

            df2["smiles"] = df2["smiles"].apply(remove_salts_and_sanitize)
            df2.dropna(subset=["smiles"], inplace=True)
            if df2.empty:
                st.error("❌ No valid SMILES after salt removal.")
                st.stop()

            st.success("✅ Salts removed.")

            # ====================================================
            # STEP 5: CHECK DISCONNECTED COMPONENTS
            # ====================================================

            disconnected = df2["smiles"].apply(check_disconnected_smiles)
            if disconnected.any():
                st.error("❌ Disconnected mixtures detected.")
                st.stop()

            st.success("✅ No disconnected components found.")

            # ====================================================
            # STEP 6: VALIDATION
            # ====================================================

            valid_mask = df2["smiles"].apply(lambda s: s is not None and is_valid_smiles(s))
            if not valid_mask.all():
                st.error("❌ Invalid SMILES detected in file.")
                st.stop()

            st.success("✅ All SMILES validated.")

            # ====================================================
            # STEP 7: STANDARDIZATION
            # ====================================================

            df2["smiles"] = Standardizer.standardize_smiles(df2["smiles"].tolist())
            st.success("✅ Standardized SMILES.")

            # ====================================================
            # STEP 8: PREPROCESS
            # ====================================================

            df2["smiles"] = df2["smiles"].apply(preprocess_molecule)
            df2.dropna(subset=["smiles"], inplace=True)
            st.success("✅ Molecules preprocessed.")

            # ====================================================
            # STEP 9: CANONICALIZE
            # ====================================================

            df2["smiles"] = df2["smiles"].apply(canonical)
            st.success("✅ Canonical SMILES generated.")

            # ====================================================
            # STEP 10: MOLD2 DESCRIPTORS
            # ====================================================

            descriptors_df = generate_mold2_descriptors(df2["smiles"])
            convert_columns_to_int64(descriptors_df)
            st.success("✅ Mold2 descriptors generated.")

            # ====================================================
            # STEP 11: PREPROCESS MODELS
            # ====================================================

            processed_car_df = preprocess_new_data_car(descriptors_df)
            processed_dili_df = preprocess_new_data_dili(descriptors_df)
            st.success("✅ CAR + DILI preprocessing complete.")

            # ====================================================
            # STEP 12: FEATURE SELECTION
            # ====================================================

            selected_car_df = processed_car_df[[c for c in important_features_car if c in processed_car_df.columns]]
            selected_dili_df = processed_dili_df[[c for c in important_features_dili if c in processed_dili_df.columns]]
            st.success("✅ Feature selection complete.")

            # ====================================================
            # STEP 13: CORAL ALIGNMENT
            # ====================================================

            selected_car_df_align = coral.transform(selected_car_df)

            selected_car_df_align = pd.DataFrame(
                selected_car_df_align,
                columns=important_features_car,
                index=selected_car_df.index
            )

            selected_car_df_align.columns = ["A" + c for c in selected_car_df_align.columns]

            selected_car_df_full = pd.concat([selected_car_df, selected_car_df_align], axis=1)
            st.success("✅ CORAL alignment complete.")

            # ====================================================
            # STEP 14: CYTOTOXIC FEATURES
            # ====================================================

            missing_features = [c for c in cytotoxic_car if c not in selected_car_df_full.columns]
            if missing_features:
                st.error(f"❌ Missing CAR features: {missing_features}")
                st.stop()

            selected_car_df_final = selected_car_df_full[cytotoxic_car]
            st.success("✅ CAR cytotoxic features selected.")

            # ====================================================
            # STEP 15: MODEL PREDICTIONS
            # ====================================================

            proba_car = calibrated_car_model.predict_proba(selected_car_df_final)
            proba_dili = calibrated_dili_model.predict_proba(selected_dili_df)
            st.success("✅ Predictions generated.")

            # ====================================================
            # STEP 16: WEIGHTED SOFT VOTING
            # ====================================================

            ensemble_proba = w_car * proba_car + w_dili * proba_dili
            ensemble_prediction = np.argmax(ensemble_proba, axis=1)
            st.success("✅ Ensemble prediction complete.")

            # ====================================================
            # STEP 17: TOXPRINT + MCP PREP
            # ====================================================

            alpha = 0.10

            idx0 = np.where(y_cal_clean == 0)[0]
            idx1 = np.where(y_cal_clean == 1)[0]

            k0 = max(1, int(np.sqrt(len(idx0))))
            k1 = max(1, int(np.sqrt(len(idx1))))

            test_fps = compute_toxprints(df2["smiles"])

            p0_list, p1_list, set_labels = [], [], []

            # ====================================================
            # STEP 18: MCP LOOP
            # ====================================================

            for i, test_fp in enumerate(test_fps):

                sims = np.array([jaccard_similarity(test_fp, fp) for fp in fp_cal_clean])

                # CLASS 0
                sims0 = sims[idx0]
                nn0_local = np.argsort(sims0)[-k0:]
                nn0 = idx0[nn0_local]

                weights0 = sims[nn0]
                weights0 = weights0 / (np.sum(weights0) + 1e-12)

                scores0 = cal_nc[nn0]
                nc0 = 1 - ensemble_proba[i, 0]
                p0 = np.sum(weights0[scores0 >= nc0])

                # CLASS 1
                sims1 = sims[idx1]
                nn1_local = np.argsort(sims1)[-k1:]
                nn1 = idx1[nn1_local]

                weights1 = sims[nn1]
                weights1 = weights1 / (np.sum(weights1) + 1e-12)

                scores1 = cal_nc[nn1]
                nc1 = 1 - ensemble_proba[i, 1]
                p1 = np.sum(weights1[scores1 >= nc1])

                # MCP SET
                pred_set = []
                if p0 > alpha: pred_set.append("0")
                if p1 > alpha: pred_set.append("1")

                set_labels.append("{" + ",".join(pred_set) + "}" if pred_set else "{}")

                p0_list.append(p0)
                p1_list.append(p1)

            # ====================================================
            # STEP 19: RESULTS
            # ====================================================

            results_df = pd.DataFrame({
                "SMILES": df2["smiles"],
                "CAR_Probability": proba_car[:, 1],
                "DILI_Probability": proba_dili[:, 1],
                "Ensemble_Probability": ensemble_proba[:, 1],
                "Prediction": np.where(ensemble_prediction == 1, "DILI Positive", "DILI Negative"),
                "MCP_Set": set_labels,
                "p_value_0": p0_list,
                "p_value_1": p1_list
            })

            # ====================================================
            # STEP 20: CONFIDENCE LABELS
            # ====================================================

            results_df["Confidence"] = np.select(
                [
                    results_df["MCP_Set"] == "{1}",
                    results_df["MCP_Set"] == "{0}",
                    results_df["MCP_Set"] == "{0,1}",
                    results_df["MCP_Set"] == "{}"
                ],
                [
                    "Confident DILI",
                    "Confident Non-DILI",
                    "Ambiguous",
                    "Evidence Failure"
                ],
                default="Unknown"
            )

            st.dataframe(results_df)
            st.sidebar.write(results_df)

            st.success("✅ Batch prediction completed.")

        except Exception as e:
            st.error(f"⚠️ Failed to process file: {e}")

    else:
        st.warning("⚠️ Please upload a CSV file.")


    
else:

    st.info(
        "ℹ️ Please input or upload SMILES data and click on 'Predict Drug Induced Hepatotoxicity'."
    )

    st.markdown("""
    <div style="border: 2px solid #4A148C; border-radius: 15px; padding: 20px; text-align: center; background-color: #f4f1fa;">
        <h5 style="color: #2d0a61;">
            This predictive framework consists of two models: an auxiliary branch (integrated CAR antagonist cytotoxicity signal)
                and a primary branch (DILI). The ensemble models were built on DILIrank data (634 training and 97 in-domain calibration compounds,
                all verified from literature). The CAR antagonist cytotoxicity signal model is independently aligned with the DILIrank dataset
                using Correlation Alignment to support hepatotoxicity prediction.
        </h5>
        <h5 style="color: white; background-color: #6A1B9A; border-radius: 10px; padding: 15px; margin-top: 20px; opacity: 0.9;">
            DILIrank hepatotoxicity predictions are more reliable when Mondrian conformal prediction indicates confidence which is probabilistic and causal in chemical space.
                however confidence is not certainty.
        </h5>
    </div>
    """, unsafe_allow_html=True)


    
    


    
    
    
    
    
    
    




