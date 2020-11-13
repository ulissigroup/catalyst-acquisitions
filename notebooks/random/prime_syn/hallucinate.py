import random
from multiprocess import Pool
from tqdm import tqdm
import ase.db

import sys
sys.path.insert(0, '../../../')
from src.discoverers.adsorption.models import PrimeModel
from src.discoverers.adsorption.randomsearch import RandomSearcher
from src.discoverers.adsorption.values import calc_co2rr_activities


# Discoverer settings
adsorbate = 'CO'
initial_training_size = 1000
batch_size = 200
quantile_cutoff = 0.9

# Data loading
db_dir = '../../pull_data/%s_synthesized/' % adsorbate
db = ase.db.connect(db_dir + '%s.db' % adsorbate)
rows = list(tqdm(db.select(), desc='reading ASE db', total=db.count()))
random.Random(42).shuffle(rows)


def parse_row(row):
    feature = row.id
    data = row.data
    label = data['adsorption_energy']
    surface = (data['mpid'], data['miller'], data['shift'], data['top'])
    return feature, label, surface


def parse_rows(rows):
    with Pool(processes=32, maxtasksperchild=1000) as pool:
        iterator = pool.imap(parse_row, rows, chunksize=100)
        iterator_tracked = tqdm(iterator, desc='parsing rows', total=len(rows))
        parsed_rows = list(iterator_tracked)

    features, labels, surfaces = list(map(list, zip(*parsed_rows)))
    return features, labels, surfaces


# Data parsing
training_features, training_labels, training_surfaces = parse_rows(rows[:initial_training_size])
sampling_features, sampling_labels, sampling_surfaces = parse_rows(rows[initial_training_size:])

# Initialize
model = PrimeModel(db_dir)
discoverer = RandomSearcher(model=model,
                            quantile_cutoff=quantile_cutoff,
                            value_calculator=calc_co2rr_activities,
                            batch_size=batch_size,
                            training_features=training_features,
                            training_labels=training_labels,
                            training_surfaces=training_surfaces,
                            sampling_features=sampling_features,
                            sampling_labels=sampling_labels,
                            sampling_surfaces=sampling_surfaces,
                            init_train=False  # Set to `False` only for warm starts
                            )

# Or load the last run
discoverer.load_last_run()

discoverer.delete_old_caches = True
discoverer.simulate_discovery()
