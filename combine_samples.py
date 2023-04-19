import numpy as np
import sys

cosmo_dir = str(sys.argv[1])


samples = []
for i in [1,2,3,4]:
    d = np.load(cosmo_dir + '/sample{}.npy'.format(i))
    samples.append(d)

np.save(cosmo_dir + '/sample.npy', np.array(samples).reshape(-1,128,128,128))
