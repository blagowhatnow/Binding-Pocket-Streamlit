import os
import streamlit as st
import py3Dmol
from stmol import showmol
import subprocess
import requests
import base64  # Import base64 for encoding binary data
import pdbfixer
from openmm import *
from openmm.app import *

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["WORLD_SIZE"] = "1"


def download_pdb(pdb_id, save_path):
    pdb_id = pdb_id.lower()  # Ensure the PDB ID is in lowercase

    # URL to download the PDB structure from the PDB website
    pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'

    try:
        # Send an HTTP GET request to download the PDB structure
        response = requests.get(pdb_url)

        if response.status_code == 200:
            # Save the downloaded content to the specified file path
            with open(save_path, 'wb') as f:
                f.write(response.content)
            st.write(f'Successfully downloaded PDB {pdb_id}')
        else:
            st.write(f'Error downloading PDB {pdb_id}. Status Code: {response.status_code}')

    except requests.exceptions.RequestException as e:
        st.write(f'Error: {e}')

def read_structure_files(directory):
    files = os.listdir(directory)
    return files

def prepare_protein(
    pdbcode, ignore_missing_residues=True, ignore_terminal_missing_residues=True, ph=7.0):
    """
    Use pdbfixer to prepare the protein from a PDB file. Hetero atoms such as ligands are
    removed and non-standard residues replaced. Missing atoms to existing residues are added.
    Missing residues are ignored by default, but can be included.

    Parameters
    ----------
    pdb_file: pathlib.Path or str
        PDB file containing the system to simulate.
    ignore_missing_residues: bool, optional
        If missing residues should be ignored or built.
    ignore_terminal_missing_residues: bool, optional
        If missing residues at the beginning and the end of a chain should be ignored or built.
    ph: float, optional
        pH value used to determine protonation state of residues

    Returns
    -------
    fixer: pdbfixer.pdbfixer.PDBFixer
        Prepared protein system.
    """
    download_pdb(pdbcode, f'{pdbcode}.pdb')
    pdb_file=str(pdbcode)+'.pdb'
    fixer = pdbfixer.PDBFixer(str(pdb_file))
    fixer.removeHeterogens(keepWater=False)  # co-crystallized ligands are unknown to PDBFixer, and removing water
    fixer.findMissingResidues()  # identify missing residues, needed for identification of missing atoms

    # if missing terminal residues shall be ignored, remove them from the dictionary
    if ignore_terminal_missing_residues:
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    # if all missing residues shall be ignored ignored, clear the dictionary
    if ignore_missing_residues:
        fixer.missingResidues = {}

  
    fixer.findNonstandardResidues()  # find non-standard residue
    fixer.replaceNonstandardResidues()  # replace non-standard residues with standard one
    fixer.findMissingAtoms()  # find missing heavy atoms
    fixer.addMissingAtoms()  # add missing atoms and residues
    fixer.addMissingHydrogens(ph)  # add missing hydrogens
    return fixer

def download_link(object_to_download, download_filename, download_link_text):

    """
    Generates a link to download the given object_to_download.

    object_to_download (str): The object to be downloaded.
    download_filename (str): Filename and extension of the file, e.g., mydata.pdb, some_text_output.pdb.
    download_link_text (str): Text to display for the download link.
    """
    # Encode the PDB data as text and provide a link for download
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title('Binding Pocket Viewer App') 
    pdb_id = st.text_input("Enter PDB ID:")
    if pdb_id:
        pdb_id = pdb_id.lower()
        if st.button("Compute Pockets"):
            #prepare protein and build only missing non-terminal residues
            prepared_protein = prepare_protein(pdb_id, ignore_missing_residues=False, ph=7.0)
            PDBFile.writeFile(prepared_protein.topology, prepared_protein.positions, open(f'{pdb_id}.pdb', 'w'))   
            subprocess.run(["python ../predict.py -p " + pdb_id+'.pdb' + " -c ../first_model_fold1_best_test_auc_85001.pth.tar -s ../seg0_best_test_IOU_91.pth.tar -r 3"], shell=True)
            subprocess.run(["mv " + pdb_id + "_nowat_pocket* out/"], shell=True)
            directory_path = 'out/'           
        # Read the structure files from the directory
            structures = read_structure_files(directory_path)  
            view = py3Dmol.view(width=800, height=500) 
        # Add the original protein structure as the first model
            with open(f'{pdb_id}.pdb') as ifile:
                protein_structure = "".join([x for x in ifile])
            view.addModel(protein_structure, "pdb")
            view.setStyle({'model': 0}, {"cartoon": {}})

        # Add all the binding pocket structures as subsequent models
            for count, structure in enumerate(structures, start=1):
                with open(os.path.join(directory_path, structure)) as ifile:
                    pocket_structure = "".join([x for x in ifile])
                view.addModel(pocket_structure, "pdb")
                view.setStyle({'model': count}, {"cartoon": {'color': 'spectrum'}})
                view.addSurface(py3Dmol.VDW,{'opacity':0.8,'colorscheme':{'prop':'b','gradient':'roygb','min':1,'max':0}}, {'model': count})                        
        # Zoom to fit all models
            view.zoomTo()
        
        # Show the Py3Dmol view
            st.header("Protein with filled in binding pockets")
            showmol(view, height=700, width=1000)
        # Display the structures using Py3Dmol
            st.header("Visualize and download individual binding pockets")
            
            count=1  #Initialize count
            for structure in structures:
                st.write("Binding Pocket "+str(count))
                with open(os.path.join(directory_path, structure)) as ifile:
                    system = "".join([x for x in ifile])       
            # Create a unique link for each binding pocket to download it
                pdb_filename = f"{pdb_id}_nowat_pocket{count}.pdb"
                with st.expander(f"Download Binding Pocket {count}"):
                # Encode the PDB data as text and provide a link for download
                    download_link_text = f"Download {pdb_filename}"
                    download_link(system, pdb_filename, download_link_text)
                xyzview = py3Dmol.view(width=800, height=500)
                xyzview.addModelsAsFrames(system)
                xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
                xyzview.addSurface(py3Dmol.SES,{'opacity':0.9,'color':'lightblue'})
                xyzview.zoomTo()
                showmol(xyzview, height=700, width=1000)
                count+=1  

            subprocess.run(["cd out/ && rm *"], shell=True)
            subprocess.run(["rm -r " + pdb_id+ '*'], shell=True)      

if __name__ == "__main__":
    main()
