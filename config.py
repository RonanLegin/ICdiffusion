import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()

  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 8
  training.n_epochs = 200001
  training.likelihood_weighting = False
  training.reduce_mean = False
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.snr = 0.075
  sampling.correct_steps = 0
  sampling.batch_size = 1
  sampling.num_samples = 25

  # data
  config.data = data = ml_collections.ConfigDict()
  data.path = '/mnt/home/rlegin/ceph/datasets/'
  data.image_size = 128
  data.num_input_channels = 2
  data.num_output_channels = 1
  data.noise_sigma = 0.1
  data.cosmo_ids = [20,40,50,70,80,90]

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 100
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.sampling_eps = 1e-5
  model.T = 1.
  model.workdir = 'run/'
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 2, 2, 1, 1)
  model.num_res_blocks = 2
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


  return config
