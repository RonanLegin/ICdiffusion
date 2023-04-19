import numpy as np
from config import get_config
import os
import logging

config = get_config()
Nside = config.data.image_size

# fiducial cosmology
if not os.path.exists(config.model.workdir + 'fiducial/'):
    os.makedirs(config.model.workdir + 'fiducial/')
    logging.warning(f"Creatig fiducial directory {config.model.workdir + 'fiducial/'}.")

input_data = np.load(config.data.path + 'fiducial/0/df_m_128_PCS_z=0.npy').reshape(-1, Nside, Nside, Nside)
label_data = np.load(config.data.path + 'fiducial/0/df_m_128_PCS_z=127.npy').reshape(-1, Nside, Nside, Nside)
label_data = (label_data - np.mean(label_data, axis=(1,2,3), keepdims=True))/np.std(label_data, axis=(1,2,3), keepdims=True)

input_data += config.data.noise_sigma * np.random.normal(size=(input_data.shape[0],Nside,Nside,Nside))
np.save(config.model.workdir + 'fiducial/observation.npy', input_data)
np.save(config.model.workdir + 'fiducial/truth.npy', label_data)


# non-fiducial cosmology
input_data = np.load(config.data.path + 'quijote128_hyper_z0_test.npy').reshape(-1, Nside, Nside, Nside)
label_data= np.load(config.data.path + 'quijote128_hyper_z127_test.npy').reshape(-1, Nside, Nside, Nside)
label_data = (label_data - np.mean(label_data, axis=(1,2,3), keepdims=True))/np.std(label_data, axis=(1,2,3), keepdims=True)
cosmo_params = np.load(config.data.path + 'quijote128_hyper_params_test.npy').reshape(-1, 5)


for cosmo_id in config.data.cosmo_ids:
	# non-fiducial cosmology
	if not os.path.exists(config.model.workdir + 'cosmo{}/'.format(cosmo_id)):
	    os.makedirs(config.model.workdir + 'cosmo{}'.format(cosmo_id))
	    logging.warning(f"Creatig fiducial directory {config.model.workdir + 'cosmo{}/'.format(cosmo_id)}.")

	# Choose example with specific cosmological parameters
	truth = label_data[cosmo_id]
	observation = input_data[cosmo_id]
	cosmo = cosmo_params[cosmo_id]
	observation += config.data.noise_sigma * np.random.normal(size=observation.shape)

	np.save(config.model.workdir + 'cosmo{}/'.format(cosmo_id) + 'observation.npy', observation)
	np.save(config.model.workdir + 'cosmo{}/'.format(cosmo_id) + 'truth.npy', truth)
	np.save(config.model.workdir + 'cosmo{}/'.format(cosmo_id) + 'cosmo.npy', cosmo)


