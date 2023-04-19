import numpy as np
from tqdm import tqdm
from config import get_config
import matplotlib.pyplot as plt
import logging
import os
import sys
import tqdm
from nbodykit.lab import ArrayMesh, FFTPower
from nbodykit import setup_logging, style
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
import matplotlib
from matplotlib.ticker import FormatStrFormatter
RdYlBu_r = matplotlib.cm.get_cmap('RdYlBu_r')
import scienceplots
plt.style.use(['science','no-latex','ieee'])

# Plotting parameters
pix_cut = 8
fs = 10
h = 3
w = 4

cosmo_dir = str(sys.argv[1]) 

config = get_config()
Nside = config.data.image_size
DEVICE = config.device

data_path = config.model.workdir + cosmo_dir


def pspec(x, boxsize=1000.0):
  mesh = ArrayMesh(x, BoxSize=boxsize)
  result = FFTPower(mesh, mode='1d', kmax=1.0)
  PS = result.power
  return PS['power'].real, PS['k']

def cross_pspec(x, y, boxsize=1000.):
  meshx = ArrayMesh(x, BoxSize=boxsize)
  meshy = ArrayMesh(y, BoxSize=boxsize)
  resultxy = FFTPower(first=meshx, mode='1d', second=meshy, kmax=1.0)
  resultxx = FFTPower(first=meshx, mode='1d', kmax=1.0)
  resultyy = FFTPower(first=meshy, mode='1d', kmax=1.0)
  PS_xy = resultxy.power['power'].real
  PS_xx = resultxx.power['power'].real
  PS_yy = resultyy.power['power'].real
  k = resultxy.power['k']
  PS = PS_xy/np.sqrt(PS_xx*PS_yy)
  return PS, k


def results(config, data_path):

  samples = np.load(os.path.join(data_path,'sample.npy')).reshape(-1,Nside,Nside,Nside)
  observation = np.load(os.path.join(data_path,'observation.npy')).reshape(-1,Nside,Nside,Nside)
  truth = np.load(os.path.join(data_path,'truth.npy')).reshape(Nside,Nside,Nside)

  if os.path.isfile(os.path.join(data_path,'/cosmo.npy')): 
    cosmo = np.load(os.path.join(data_path,'/cosmo.npy')).reshape(5)
  else:
    cosmo = None
  
  # Compute correlation as function of density threshold in observation
  max_amp = np.max(observation)
  min_amp = np.min(observation)

  threshold = np.linspace(min_amp, max_amp, 64)

  corr_coefs = []
  # Compute Pearson's correlation coefficient
  for sample in samples:
      rho_set = []
      for thld in threshold:
        X = sample[observation[0] < thld].reshape(-1)
        Y = truth[observation[0] < thld].reshape(-1)
        #rho = np.cov(X,Y)/(np.std(X)*np.std(Y))
        rho = (np.mean(X*Y) - np.mean(X)*np.mean(Y))/(np.std(X)*np.std(Y))
        rho_set.append(rho)
      corr_coefs.append(np.array(rho_set))

  mean_corr = np.mean(corr_coefs, axis=0)
  std_corr = np.std(corr_coefs, axis=0)

  # Plot power spectra of truth vs generated samples
  plt.figure(figsize=(w,h))
  if cosmo is not None:
    plt.title(r'$\omega_m = {:.4f}, \omega_b = {:.4f}, h = {:.4f}, n_s = {:.4f}, \sigma_8 = {:.4f}$'.format(
      cosmo[0],
      cosmo[1],
      cosmo[2],
      cosmo[3],
      cosmo[4]
      ),
    fontsize=fs
    )
  plt.plot(threshold, mean_corr, color='r')
  plt.fill_between(threshold, mean_corr - std_corr, mean_corr+std_corr, alpha=0.2, color='#82A8D1')
  plt.axhline(1.0, color='k', lw='2')
  #plt.xscale('log')
  #plt.yscale('log')
  plt.xlabel(r"$Density Threshold$", fontsize=fs)
  plt.ylabel(r"$\rho$", fontsize=fs)
  plt.legend()    
  plt.savefig(os.path.join(data_path,'corr_coef.pdf'), bbox_inches='tight')#, dpi=200)



  # Compute power spectrum of true initial condition
  truth_pspec, truth_k = pspec(truth)
  truth_pspec = truth_pspec[1:]
  truth_k = truth_k[1:]

  samples_crosspspec = []
  samples_pspec = []
  samples_k = []

  for i, sample in enumerate(tqdm.tqdm(samples, total=samples.shape[0], desc='computing power spectra')):
      ps, k = pspec(sample)
      ps_cross, k_cross = cross_pspec(sample, truth)
      samples_crosspspec.append(ps_cross)
      samples_pspec.append(ps)
      samples_k.append(k)


  samples_pspec = np.array(samples_pspec)[:,1:]
  samples_k = np.array(samples_k)[:,1:]
  mean_pspec = np.mean(samples_pspec, axis=0)
  std_pspec = np.std(samples_pspec, axis=0)

  samples_crosspspec = np.array(samples_crosspspec)[:,1:]
  mean_crossps = np.mean(samples_crosspspec, axis=0)
  std_crossps = np.std(samples_crosspspec, axis=0)

  # Save power spectrum of generated samples
  np.save(os.path.join(data_path,'pspec.npy'), samples_pspec)
  np.save(os.path.join(data_path,'k.npy'), samples_pspec)


  tf_set = []
  for i in range(samples_pspec.shape[0]):
    tf_set.append(np.sqrt(samples_pspec[i]/truth_pspec))
  tf_set = np.array(tf_set)
  mean_tf = np.mean(tf_set, axis=0)
  std_tf = np.std(tf_set, axis=0)

  # Save transfer function of generated samples
  np.save(os.path.join(data_path,'tf.npy'), tf_set)


  fig, axs = plt.subplots(3, sharex=True, sharey=False, height_ratios=[2, 1, 1])
  # Plot power spectra of truth vs generated samples
  fig.set_size_inches((w, h*2)) 
  axs[0].plot(samples_k[-1], mean_pspec, color='#82A8D1', label='Inferred')
  axs[0].fill_between(samples_k[-1], mean_pspec - 2*std_pspec, mean_pspec+2*std_pspec, alpha=0.5, color='#82A8D1')
  axs[0].plot(truth_k, truth_pspec, color='k', ls='--', lw=1, label='Truth')
  axs[0].set_xscale('log')
  axs[0].set_yscale('log')
  axs[0].tick_params(axis='x', which='both',length=0)
  #if np.sum(cosmo_idx == np.array([50,70,80,40])) == 0:
  axs[0].set_ylabel(r"$P(k)$")
  axs[0].legend()
  if cosmo is not None:    
    textstr = '\n'.join((
      r'$\Omega_m=%.4f$' % (cosmo[0], ),
      r'$\Omega_b=%.4f$' % (cosmo[1], ),
      r'$h=%.4f$' % (cosmo[2], ),
      r'$n_s=%.4f$' % (cosmo[3], ),
      r'$\sigma_8=%.4f$' % (cosmo[4], )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    axs[0].text(0.05, 0.45, textstr, transform=axs[0].transAxes, fontsize=fs,
            verticalalignment='top', bbox=props)
  axs[0].set_xlim(left=samples_k[-1,0])

  # Plot cross-correlation of true vs samples
  axs[1].plot(samples_k[-1], mean_crossps, color='#82A8D1')
  axs[1].fill_between(samples_k[-1], mean_crossps - 2*std_crossps, mean_crossps+2*std_crossps, alpha=0.5, color='#82A8D1')
  axs[1].axhline(1.0, color='k', ls='--', lw=1)
  axs[1].set_xscale('log')
  axs[1].tick_params(axis='x', which='both',length=0)
  axs[1].set_ylabel(r"$C(k)$")
  axs[1].set_xlim(left=samples_k[-1,0])
  axs[1].legend()    

  # Plot transfer function of sample
  axs[2].plot(samples_k[-1], mean_tf, color='#82A8D1')
  axs[2].fill_between(samples_k[-1], mean_tf - 2*std_tf, mean_tf+2*std_tf, alpha=0.5, color='#82A8D1')
  axs[2].axhline(1.0, color='k', ls='--', lw=1)
  axs[2].set_xscale('log')
  axs[2].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
  axs[2].set_ylabel(r"$T(k)$")
  axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  axs[2].set_ylim(bottom=0.95,top=1.1)
  if np.max(mean_tf) < 1.0:
    axs[2].set_ylim(top=1.005)
  axs[2].set_xlim(left=samples_k[-1,0])
  axs[2].legend()    
  plt.subplots_adjust(hspace=0)
  plt.savefig(os.path.join(data_path,'pspec.pdf'))#, dpi=200)



  # Quantify accuracy
  mu = np.mean(samples, axis=0)
  sigma = np.std(samples, axis=0)
  norm_truth = (truth - mu)/sigma

  # Plot distribution of normalized true initial condition
  fig, ax = plt.subplots(figsize=(w,h))
  #plt.title('Fiducial Cosmology', fontsize=fs)
  if cosmo is not None:
    ax.set_title(r'$\omega_m: {:.4f}, \omega_b: {:.4f}, h: {:.4f}, n_s: {:.4f}, \sigma_8: {:.4f}$'.format(
      cosmo[0],
      cosmo[1],
      cosmo[2],
      cosmo[3],
      cosmo[4]
      ),
    fontsize=fs
    )
  ax.hist(norm_truth.flatten(), bins=100, density=True, color='#82A8D1')
  x = np.linspace(0. - 5*1.0, 0. + 5*1.0, 500)
  ax.plot(x, stats.norm.pdf(x, 0., 1.0), linestyle='--',color='k', lw=1, label='Normal Distribution')
  plt.ylabel('Probability Density', fontsize=fs)
  #ax.get_yaxis().set_visible(False)
  ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
  ax.set_xlabel(r'$z_{\text{true}}$', fontsize=fs)
  ax.set_xlim([-7., 7.])
  ax.legend()
  #plt.savefig(os.path.join(data_path, 'coverage.pdf'), bbox_inches='tight')#, dpi=200)


  # Plot obs-sample-true
  fig = plt.figure(figsize=(3*w,h+1))
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)
  ax3 = fig.add_subplot(133)

  img1 = ax1.imshow(np.mean(observation[0,:,:,:pix_cut], axis=2), extent=(0, 1000, 0, 1000), cmap='inferno')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img1, cax=cax)
  ax1.set_title('Present-Day z = 0', fontsize=fs)
  ax1.tick_params(axis='both', which='both',length=0)
  ax1.get_xaxis().set_visible(False)
  ax1.get_yaxis().set_visible(False)
  img2 = ax2.imshow(np.mean(samples[0,:,:,:pix_cut], axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img2, cax=cax)
  ax2.set_title('Predicted z = 127', fontsize=fs)
  ax2.tick_params(axis='both', which='both',length=0)
  ax2.get_xaxis().set_visible(False)
  ax2.get_yaxis().set_visible(False)
  img3 = ax3.imshow(np.mean(truth[:,:,:pix_cut],axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax3)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img3, cax=cax)
  ax3.set_title('True z = 127', fontsize=fs)
  ax3.tick_params(axis='both', which='both',length=0)
  ax3.get_xaxis().set_visible(False)
  ax3.get_yaxis().set_visible(False)
  plt.subplots_adjust(wspace=0.125)
  plt.savefig(os.path.join(data_path,'obs_sample_true.pdf'))#, dpi=400)

  # Plot obs-true-sample_mean-sample_std
  fig = plt.figure(figsize=(4*w,h))
  ax1 = fig.add_subplot(141)
  ax2 = fig.add_subplot(142)
  ax3 = fig.add_subplot(143)
  ax4 = fig.add_subplot(144)

  img1 = ax1.imshow(np.mean(observation[0,:,:,:pix_cut], axis=2), extent=(0, 1000, 0, 1000), cmap='inferno')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img1, cax=cax)
  ax1.set_title('Present-Day z = 0', fontsize=fs)
  img2 = ax2.imshow(np.mean(truth[:,:,:pix_cut],axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img2, cax=cax)
  ax2.set_title('True z = 127', fontsize=fs)
  img3 = ax3.imshow(np.mean(np.mean(samples[:,:,:,:pix_cut],axis=0), axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax3)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img3, cax=cax)
  ax3.set_title('Mean', fontsize=fs)
  img4 = ax4.imshow(np.mean(np.std(samples[:,:,:,:pix_cut],axis=0)**2, axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax4)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img4, cax=cax)
  ax4.set_title('Variance',fontsize=fs)
  plt.savefig(os.path.join(data_path, 'obs_true_mean_var.pdf'), bbox_inches='tight')#, dpi=200)

  # Plot true-sample_mean-sample_std
  fig = plt.figure(figsize=(3*w,h))
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)
  ax3 = fig.add_subplot(133)

  img1 = ax1.imshow(np.mean(truth[:,:,:pix_cut],axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img1, cax=cax)
  ax1.set_title('Ground Truth', fontsize=fs)
  img2 = ax2.imshow(np.mean(np.mean(samples[:,:,:,:pix_cut],axis=0), axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img2, cax=cax)
  ax2.set_title('Predicted Mean', fontsize=fs)
  img3 = ax3.imshow(np.mean(np.std(samples[:,:,:,:pix_cut],axis=0)**2, axis=2), extent=(0, 1000, 0, 1000), cmap='RdYlBu')
  divider = make_axes_locatable(ax3)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img3, cax=cax)
  ax3.set_title('Predicted Variance', fontsize=fs)
  plt.savefig(os.path.join(data_path,'true_mean_var.pdf'), bbox_inches='tight')#, dpi=200)


  # Plot true-sample_mean-sample_std
  fig = plt.figure(figsize=(2*w,h))
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)

  img1 = ax1.imshow(np.mean(truth[:,:,:pix_cut],axis=2), extent=(0, 1000, 0, 1000), cmap='inferno')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img1, cax=cax)
  ax1.set_title('Present-day z = 0', fontsize=fs)
  img2 = ax2.imshow(np.mean(np.std(samples[:,:,:,:pix_cut],axis=0)**2, axis=2), extent=(0, 1000, 0, 1000), cmap='inferno')
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img2, cax=cax)
  ax2.set_title('Sample Variance z = 127', fontsize=fs)
  plt.savefig(os.path.join(data_path,'obs_var.pdf'), bbox_inches='tight')#, dpi=200)


  # Plot true-sample_mean-sample_std
  fig = plt.figure(figsize=(w,h))
  ax1 = fig.add_subplot(131)
  img1 = ax1.imshow(np.mean(np.std(samples[:,:,:,:pix_cut],axis=0)**2, axis=2), extent=(0, 1000, 0, 1000), cmap='inferno')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(img1, cax=cax)
  plt.savefig(os.path.join(data_path,'var.pdf'), bbox_inches='tight')#, dpi=200)


results(config,data_path)
