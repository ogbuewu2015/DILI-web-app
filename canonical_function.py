# canonical_function.py

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#importing the cheminformatics module needed
import molvs
from molvs import Standardizer, normalize
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import rdkit
print(rdkit.__version__)
from rdkit import Chem

def canonical(smiles):

    # Handle None
    if smiles is None:
        return None

    # Handle NaN / float values
    if isinstance(smiles, float):
        return None

    # Convert to string safely
    smiles = str(smiles).strip()

    # Empty string check
    if smiles == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        if mol is not None:
            # Convert to canonical SMILES
            canonical_smiles = Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                canonical=True
            )
            return canonical_smiles

        else:
            print(f"Unable to parse SMILES: {smiles}")
            return None

    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(e)
        return None
