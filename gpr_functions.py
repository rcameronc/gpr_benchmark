import glob
from matplotlib.colors import Normalize
import pandas as pd
import scipy.io as io
import xarray as xr
import numpy as np

from itertools import product

import tensorflow as tf
from tensorflow_probability import bijectors as tfb

# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# from pandas.io.json import json_normalize
# from df2gspread import df2gspread as d2g


import gpflow
import gpflow as gpf
from gpflow.utilities import print_summary, positive
from gpflow.logdensities import multivariate_normal
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, RegressionData, MeanAndVariance, GPModel
from gpflow.base import Parameter 
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.config import default_jitter, default_float
from gpflow.ci_utils import ci_niter, ci_range


from typing import Optional, Tuple

from matplotlib import colors, cm

from gpflow.inducing_variables import InducingPoints


def make_mod(path, ice_model, lith, ages, extent, zeros=False):
        
    """combine model runs from local directory into xarray dataset."""
    
    files = f'{path}*.nc'
    basefiles = glob.glob(files)
    modelrun = [key.split('output_', 1)[1][:-3] for key in basefiles]
    dss = xr.open_mfdataset(files,
                            chunks=None,
                            concat_dim='modelrun',
                            combine='nested')
    lats, lons, times = dss.LAT.values[0], dss.LON.values[0], dss.TIME.values[0]
    ds = dss.drop(['LAT', 'LON', 'TIME']).assign_coords(lat=lats,
                                                        lon=lons,
                                                        time=times * 1000,
                                                        modelrun=modelrun).rename({
                                                                      'time': 'age', 'RSL': 'rsl'})
    ds = ds.chunk({'lat': 10, 'lon': 10})
    ds = ds.roll(lon=256, roll_coords=True)
    ds.coords['lon'] = pd.DataFrame((ds.lon[ds.lon >= 180] - 360)- 0.12 ) \
                                    .append(pd.DataFrame(ds.lon[ds.lon < 180]) + 0.58) \
                                    .reset_index(drop=True).squeeze()
    ds = ds.swap_dims({'dim_0': 'lon'}).drop('dim_0')
    
    #slice dataset to location
    ds = ds.rsl.sel(age=slice(ages[0], ages[-1]),
            lon=slice(extent[0] - 2, extent[1] + 2),
            lat=slice(extent[3] + 2, extent[2] - 2))
    
    if zeros:
    #add present-day RSL at zero to the GIA model
        ds_zeros = xr.zeros_like(ds)[:,0] + 0.01
        ds_zeros['age'] = 0.1
        ds_zeros = ds_zeros.expand_dims('age').transpose('modelrun','age', 'lon', 'lat')
        ds = xr.concat([ds, ds_zeros], 'age')
    else:
        pass
    
    return ds


def add_presday_0s(df_place, nout):
    
    """ Prescribe present-day RSL to zero by adding zero points at t=10 yrs."""
    
    #prescribe present-day RSL to zero
    preslocs1 = df_place.groupby(['lat', 'lon'])[['rsl', 
                                                 'rsl_er_max',
                                                 'age']].nunique().reset_index()[['lat',
                                                                                  'lon']][::int(50/nout)]

    # make more present day points at zero on an nout/nout grid
    lat = np.linspace(min(df_place.lat), max(df_place.lat), nout)
    lon = np.linspace(min(df_place.lon), max(df_place.lon), nout)
    xy = np.array(list(product(lon, lat)))[::int(nout/2)]

    preslocs2 = pd.DataFrame(xy, columns=['lon', 'lat'])

    preslocs = pd.concat([preslocs1, preslocs2]).reset_index(drop=True)

    preslocs['rsl'] = 0.1
    preslocs['rsl_er'] = 0.1
    preslocs['rsl_er_max'] = 0.1
    preslocs['rsl_er_min'] = 0.1
    preslocs['age_er'] = 1
    preslocs['age_er_max'] = 1
    preslocs['age_er_min'] = 1
    preslocs['age'] = 10

    df_place = pd.concat([df_place, preslocs]).reset_index(drop=True)
    return df_place


def import_rsls(path, tmin, tmax, extent):
    
    """ import khan Holocene RSL database from csv."""
    
    df = pd.read_csv(path, encoding="ISO-8859-15", engine='python')
    df = df.replace('\s+', '_', regex=True).replace('-', '_', regex=True).\
            applymap(lambda s:s.lower() if type(s) == str else s)
    df.columns = df.columns.str.lower()
    df.rename_axis('index', inplace=True)
    df = df.rename({'latitude': 'lat', 'longitude': 'lon'}, axis='columns')
    dfind, dfterr, dfmar = df[(df.type == 0)
                              & (df.age > 0)], df[df.type == 1], df[df.type == -1]

    #select location
    df_slice = dfind[(dfind.age > tmin) & (dfind.age < tmax) &
                     (dfind.lon > extent[0])
                     & (dfind.lon < extent[1])
                     & (dfind.lat > extent[2])
                     & (dfind.lat < extent[3])][[
                        'lat', 'lon', 'rsl', 'rsl_er_max', 'rsl_er_min', 'age', 'age_er_max', 'age_er_min']]
    df_slice['rsl_er'] = (df_slice.rsl_er_max + df_slice.rsl_er_min)/2
    df_slice['age_er'] = (df_slice.age_er_max + df_slice.age_er_min)/2
    df_place = df_slice.copy()
    return df_place

def xarray_template(filename, ages, extent):
    
    """make template for xarray interpolation"""

    template = io.loadmat(filename, squeeze_me=True)

    template = xr.Dataset({'rsl': (['lat', 'lon', 'age'], np.zeros((256, 512, len(ages))))},
                         coords={'lon': template['lon_out'],
                                 'lat': template['lat_out'],
                                 'age': ages})
    template.coords['lon'] = pd.DataFrame((template.lon[template.lon >= 180] - 360)- 0.12) \
                            .append(pd.DataFrame(template.lon[template.lon < 180]) + 0.58) \
                            .reset_index(drop=True).squeeze()
    ds_template = template.swap_dims({'dim_0': 'lon'}).drop('dim_0').sel(lon=slice(extent[0] + 180 - 2, 
                                                                                   extent[1] + 180 + 2),
                                                                         lat=slice(extent[3] + 2,
                                                                                   extent[2] - 2)).rsl
    return ds_template


def cmap_codes(name, number):
    
    " make colormap hexcodes"
    
    cmap = cm.get_cmap(name, number) 
    hexcodes = []
    for i in range(cmap.N): 
        hexcodes.append(colors.rgb2hex(cmap(i)[:3]))
    return hexcodes


class MidpointNormalize(Normalize):
    
    """Normalise the colorbar.  e.g. norm=MidpointNormalize(mymin, mymax, 0.)"""
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
    
def normalize(df):
    return np.array((df - df.mean()) / df.std()).reshape(len(df), 1)



def denormalize(y_pred, df):
    return np.array((y_pred * df.std()) + df.mean())



def bounded_parameter(low, high, param):
    """Make parameter tfp Parameter with optimization bounds."""

    sigmoid = tfb.Sigmoid(low=tf.cast(low, tf.float64), 
                          high=tf.cast(high, tf.float64),
                         name='sigmoid')
    parameter = gpf.Parameter(param, transform=sigmoid, dtype=tf.float64)
    return parameter


class HaversineKernel_Matern32(gpf.kernels.Matern32):
    """
    Isotropic Matern52 Kernel with Haversine distance instead of euclidean distance.
    Assumes n dimensional data, with columns [latitude, longitude] in degrees.
    """
    def __init__(
        self,
        lengthscales=None,
        variance=1.0,
        active_dims=None,
    ):
        super().__init__(
            active_dims=active_dims,
            variance=variance,
            lengthscales=lengthscales,

        )

    def haversine_dist(self, X, X2):
        pi = np.pi / 180
        f = tf.expand_dims(X * pi, -2)  # ... x N x 1 x D
        f2 = tf.expand_dims(X2 * pi, -3)  # ... x 1 x M x D
        d = tf.sin((f - f2) / 2)**2
        lat1, lat2 = tf.expand_dims(X[:, 0] * pi, -1), \
                    tf.expand_dims(X2[:, 0] * pi, -2)
        cos_prod = tf.cos(lat2) * tf.cos(lat1)
        a = d[:, :, 0] + cos_prod * d[:, :, 1]
        c = tf.asin(tf.sqrt(a)) * 6371 * 2
        return c

    def scaled_squared_euclid_dist(self, X, X2):
        """
        Returns (dist(X, X2ᵀ)/lengthscales)².
        """
        if X2 is None:
            X2 = X
        dist = tf.square(self.haversine_dist(X, X2) / self.lengthscales)

        return dist
    
    
class GPR_new(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by
    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
#         noise_variance: float = 1.0,
        noise_variance: list = [],
    ):
        
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        K = self.kernel(X)
        num_data = X.shape[0]
        k_diag = tf.linalg.diag_part(K)
#         s_diag = tf.fill([num_data], self.likelihood.variance)
        s_diag = tf.convert_to_tensor(self.likelihood.variance)

        
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]

#         s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))
        s = tf.linalg.diag(tf.convert_to_tensor(self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

    
def locs_with_enoughsamples(df_place, place, number):
    """make new dataframe, labeled, of sites with [> number] measurements"""
    df_lots = df_place.groupby(['lat',
                                'lon']).filter(lambda x: len(x) > number)

    df_locs = []
    for i, group in enumerate(df_lots.groupby(['lat', 'lon'])):
        singleloc = group[1].copy()
        singleloc['location'] = place
        singleloc['locnum'] = place + '_site' + str(
            i)  # + singleloc.reset_index().index.astype('str')
        df_locs.append(singleloc)
    df_locs = pd.concat(df_locs)

    return df_locs

def ds_select(ds, row):
    
    """ Slice GIA or GPR xarray dataset by rows in RSL pandas dataframe for rsl."""
    
    return ds.rsl.sel(age=[row.age],
                      lon=[row.lon],
                      lat=[row.lat],
                      method='nearest').squeeze().values


def ds_ageselect(ds, row):
    
    """ Slice GIA or GPR xarray dataset by rows in RSL pandas dataframe for age."""
    
    return ds.rsl.interp(age=[row.age]).age.values[0]


def interp_likegpr(ds_single, da_zp):
    return ds_single.load().interp_like(da_zp)


class GPR_new(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by
    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: list = [],
    ):
        
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        K = self.kernel(X)
        num_data = X.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.convert_to_tensor(self.likelihood.variance)
        
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]

        s = tf.linalg.diag(tf.convert_to_tensor(self.likelihood.variance))
        
        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False)  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
def predict_post_f(nout, ages, ds_single, df_place, m):

    # make variables
    lat = np.linspace(min(ds_single.lat), max(ds_single.lat), nout)
    lon = np.linspace(min(ds_single.lon), max(ds_single.lon), nout)
    xyt = np.array(list(product(lon, lat, ages)))

    # predict posterior RSL mean & variance
    #iteration reduces memory pressure of haversine calculation
    # Couldn't figure out how to make this pythonic 
    y_preds = []
    var = []
    for i in range(len(ages)):
        y_pred, var_it = m.predict_f(xyt[:nout**2])
        xyt = xyt[nout**2:]
        y_preds.append(y_pred)
        var.append(var_it)
    y_pred = tf.concat(y_preds, 0)  
    var = tf.concat(var, 0)
    
    #denormalize to return correct values
    y_pred = denormalize(y_pred, df_place.rsl_realresid)
    
    # reshape output vectors
    Zp = np.array(y_pred).reshape(nout, nout, len(ages))
    varp = np.array(var).reshape(nout, nout, len(ages))
    
    #transform output into xarray dataarrays
    da_zp = xr.DataArray(Zp, coords=[lon, lat, ages],
                     dims=['lon', 'lat','age']).transpose('age', 'lat', 'lon')
    da_varp = xr.DataArray(varp, coords=[lon, lat, ages],
                     dims=['lon', 'lat','age']).transpose('age', 'lat', 'lon')
    
    return da_zp, da_varp   
    
    
def run_gpr(nout, iterations, ds_single, ages, k1len, k2len, k3len, k4len, df_place):
            
            
    # Input space, rsl normalized to zero mean, unit variance
    X = np.stack((df_place.lon, df_place.lat, df_place.age), 1)

    RSL = normalize(df_place.rsl_realresid)
    
    #define kernels  with bounds
    k1 = gpf.kernels.Matern32(active_dims=[0, 1])
    k1.lengthscales = bounded_parameter(1, 10, k1len) 
    k1.variance = bounded_parameter(0.02, 100, 2)

    k2 = gpf.kernels.Matern32(active_dims=[2])
    k2.lengthscales = bounded_parameter(1, 100000, k2len)
    k2.variance = bounded_parameter(0.02, 100, 1)

    k3 = gpf.kernels.Matern32(active_dims=[0, 1])
    k3.lengthscales = bounded_parameter(10, 100, k3len) 
    k3.variance = bounded_parameter(0.01, 100, 1)

    k4 = gpf.kernels.Matern32(active_dims=[2]) 
    k4.lengthscales = bounded_parameter(1, 100000, k4len)
    k4.variance = bounded_parameter(0.01, 100, 1)

    k5 = gpf.kernels.White(active_dims=[0, 1, 2])
    k5.variance = bounded_parameter(0.01, 100, 1)

    kernel = (k1 * k2) + (k3 * k4) + k5 

    ##################	  BUILD AND TRAIN MODELS 	#######################
    noise_variance = (df_place.rsl_er.ravel())**2  

    m = GPR_new((X, RSL), kernel=kernel, noise_variance=noise_variance) 
    
    #Sandwich age of each lat/lon to enable gradient calculation
    lonlat = df_place[['lon', 'lat']]
    agetile = np.stack([df_place.age - 10, df_place.age, df_place.age + 10], axis=-1).flatten()
    xyt_it = np.column_stack([lonlat.loc[lonlat.index.repeat(3)], agetile])

    #hardcode indices for speed (softcoded alternative commented out)
    indices = np.arange(1, len(df_place)*3, 3)
    # indices = np.where(np.in1d(df_place.age, agetile))[0]
    
    iterations = ci_niter(iterations)
    learning_rate = 0.05
    logging_freq = 100
    opt = tf.optimizers.Adam(learning_rate)

    #first optimize without age errs to get slope
    tf.print('___First optimization___')
    likelihood = -10000
    for i in range(iterations):
        opt.minimize(m.training_loss, var_list=m.trainable_variables)

        likelihood_new = m.log_marginal_likelihood()
        if i % logging_freq == 0:
            tf.print(f"iteration {i + 1} likelihood {m.log_marginal_likelihood():.04f}")
            if abs(likelihood_new - likelihood) < 0.001:
                    break
        likelihood = likelihood_new

    # Calculate posterior at training points + adjacent age points
    mean, _ = m.predict_f(xyt_it)

    # make diagonal matrix of age slope at training points
    Xgrad = np.diag(np.gradient(mean.numpy(), axis=0)[indices][:,0])

    # multipy age errors by gradient 
    Xnigp = np.diag(Xgrad @ np.diag((df_place.age_er/2)**2) @ Xgrad.T)    
    
    m = GPR_new((X, RSL), kernel=kernel, noise_variance=noise_variance + Xnigp)

    #reoptimize
    tf.print('___Second optimization___')
    opt = tf.optimizers.Adam(learning_rate)
    
    for i in range(iterations):
        opt.minimize(m.training_loss, var_list=m.trainable_variables)
        
        likelihood_new = m.log_marginal_likelihood()
        if i % logging_freq == 0:
            tf.print(f"iteration {i + 1} likelihood {m.log_marginal_likelihood():.04f}")
            if abs(likelihood_new - likelihood) < 0.001:
                    break
        likelihood = likelihood_new
            
    ##################	  INTERPOLATE MODELS 	#######################
    ##################  --------------------	 ######################
    # output space
    da_zp, da_varp = predict_post_f(nout, ages, ds_single, df_place, m)

    #interpolate all models onto GPR grid
    ds_giapriorinterp  = interp_likegpr(ds_single, da_zp)

    # add total prior RSL back into GPR
    ds_priorplusgpr = da_zp + ds_giapriorinterp
    ds_varp = da_varp.to_dataset(name='rsl')
    ds_zp = da_zp.to_dataset(name='rsl')

    #Calculate data-model misfits & GPR vals at RSL data locations
    df_place['gpr_posterior'] = df_place.apply(lambda row: ds_select(ds_priorplusgpr, row), axis=1)
    df_place['gprpost_std'] = df_place.apply(lambda row: ds_select(ds_varp, row), axis=1)
    df_place['gpr_diff'] = df_place.apply(lambda row: row.rsl - row.gpr_posterior, axis=1)
    df_place['diffdiv'] = df_place.gpr_diff / df_place.rsl_er
    
    k1_l = m.kernel.kernels[0].kernels[0].lengthscales.numpy()
    k2_l = m.kernel.kernels[0].kernels[1].lengthscales.numpy()
    k3_l = m.kernel.kernels[1].kernels[0].lengthscales.numpy()
    k4_l = m.kernel.kernels[1].kernels[1].lengthscales.numpy()
    
    
    return ds_giapriorinterp, ds_zp, ds_priorplusgpr, ds_varp, m.log_marginal_likelihood().numpy(), m, df_place

    def interp_ds(ds):
        return ds.interp(age=ds_single.age, lat=ds_single.lat, lon=ds_single.lon)

    def slice_dataset(ds):
        return ds.rsl.sel(lat=site[1].lat.unique() ,
                      lon=site[1].lon.unique(),
                      method='nearest').sel(age=slice(11500, 0))
    
    def ds_ageselect(ds, row):
        return ds.rsl.interp(age=[row.age]).age.values[0]