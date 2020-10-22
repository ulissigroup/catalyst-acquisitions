import os
from collections import defaultdict
import random
import pickle
from scipy.stats import norm
from tqdm import tqdm
import ase.db
from gaspy.gasdb import get_surface_from_doc, get_mongo_collection
from gaspy.mongo import make_atoms_from_doc


# Load bulk data
with open('sites_by_mpid.pkl', 'rb') as file_handle:
    sites_by_mpid = pickle.load(file_handle)
catalog_docs = [doc for docs in sites_by_mpid.values() for doc in docs]

# Organize sites by bulk
counts_by_mpid = {mpid: len(sites) for mpid, sites in sites_by_mpid.items()}

# Organize sites by surface
sites_by_surface = defaultdict(list)
for doc in tqdm(catalog_docs, unit_scale=True, desc='organizing sites'):
    surface = get_surface_from_doc(doc)
    sites_by_surface[surface].append(doc)

counts_by_surface = {surface: len(sites) for surface, sites in sites_by_surface.items()}

# Organize surfaces by bulk
surfaces_by_bulk = defaultdict(list)
for surface, sites in tqdm(sites_by_surface.items(), unit_scale=True):
    bulk = surface[0]
    surfaces_by_bulk[bulk].append(surface)

surface_count_by_bulk = {bulk: len(surfaces) for bulk, surfaces in surfaces_by_bulk.items()}


# How many bulks we want to sample
n_bulks = 200

# Choose bulks & sites
samples = {}
mpids = list(counts_by_mpid.keys())
random.shuffle(mpids)
for mpid in tqdm(mpids[:n_bulks]):
    surfaces = surfaces_by_bulk[mpid]
    _sites_by_surface = {surface: sites_by_surface[surface] for surface in surfaces}
    samples[mpid] = _sites_by_surface

n_sites = sum([1 for _sites_by_surface in samples.values() for sites in _sites_by_surface.values() for site in sites])
print(n_sites)

# Shape of synthesized data
bulk_mean = 0.
bulk_std = 1.
surface_std = 0.3
site_std = 0.1

# Initialize synthesized dataset
db_name = 'CO.db'
try:
    os.remove(db_name)
except FileNotFoundError:
    pass
db = ase.db.connect(db_name)

# Grab all the sites from chosen bulks
mongo_ids = [site['mongo_id']
             for bulks in samples.values()
             for surfaces in bulks.values()
             for site in surfaces]
query = {'_id': {'$in': mongo_ids}}
projection = {'atoms': 1, 'calc': 1, 'results': 1}
with get_mongo_collection('catalog') as collection:
    all_docs = list(tqdm(collection.find(query, projection), desc='pulling docs', total=n_sites))
docs_by_id = {doc['_id']: doc for doc in all_docs}

# Make up an energy
for mpid, _samples in tqdm(samples.items(), desc='bulks'):
    sampled_bulk_mean = norm.rvs(loc=bulk_mean, scale=bulk_std)
    for surface, sites in tqdm(_samples.items(), desc='surfaces'):
        sampled_surface_mean = norm.rvs(loc=sampled_bulk_mean, scale=surface_std)
        for site in tqdm(sites, desc='sites'):
            sampled_energy = norm.rvs(loc=sampled_surface_mean, scale=site_std)

            # Make the atoms object
            doc = docs_by_id[site['mongo_id']]
            atoms = make_atoms_from_doc(doc)

            # Grab meta info
            miller = tuple(site['miller'])
            shift = round(site['shift'], 2)
            top = site['top']
            data = {'adsorption_energy': sampled_energy,
                    'mpid': mpid,
                    'miller': miller,
                    'shift': shift,
                    'top': top}

            # Write to the DB
            db.write(atoms, data=data)
