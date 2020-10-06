import random
import ase.db

import sys
sys.path.insert(0, '../../../')
from src.discoverers.adsorption.values import calc_co2rr_activities
from src.discoverers.adsorption.mms import MultiscaleDiscoverer
from src.discoverers.adsorption.models import PrimeModel


# Discoverer settings
adsorbate = 'CO'
initial_training_size = 1000
batch_size = 200
quantile_cutoff = 0.95

# Data loading
db_dir = '../../pull_data/%s_truncated/' % adsorbate
db = ase.db.connect(db_dir + '%s.db' % adsorbate)
rows = list(db.select())
random.Random(42).shuffle(rows)


def parse_rows(rows):
    features = []
    labels = []
    surfaces = []

    for row in rows:
        features.append(row.id)
        data = row.data
        labels.append(data['adsorption_energy'])
        surface = (data['mpid'], data['miller'], data['shift'], data['top'])
        surfaces.append(surface)

    return features, labels, surfaces


# Data parsing
training_features, training_labels, training_surfaces = parse_rows(rows[:initial_training_size])
sampling_features, sampling_labels, sampling_surfaces = parse_rows(rows[initial_training_size:])

# Initialize
model = PrimeModel(db_dir)
discoverer = MultiscaleDiscoverer(model=model,
                                  quantile_cutoff=quantile_cutoff,
                                  value_calculator=calc_co2rr_activities,
                                  batch_size=batch_size,
                                  training_features=training_features,
                                  training_labels=training_labels,
                                  training_surfaces=training_surfaces,
                                  sampling_features=sampling_features,
                                  sampling_labels=sampling_labels,
                                  sampling_surfaces=sampling_surfaces,
                                  #init_train=False  # Set to `False` only for warm starts
                                  )

#discoverer.load_last_run()

discoverer.simulate_discovery()
