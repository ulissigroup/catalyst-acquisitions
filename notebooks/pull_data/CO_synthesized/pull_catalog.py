import pickle
from tqdm import tqdm
from gaspy.gasdb import get_catalog_docs


docs = get_catalog_docs()

mpids = {doc['mpid'] for doc in tqdm(docs, desc='finding mpids')}
sites_by_mpid = {mpid: [] for mpid in mpids}

for doc in tqdm(docs, desc='sorting sites'):
    sites_by_mpid[doc['mpid']].append(doc)

with open('sites_by_mpid.pkl', 'wb') as file_handle:
    pickle.dump(sites_by_mpid, file_handle)
