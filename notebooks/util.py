import os

import time

import intake
import numpy as np
import xarray as xr

import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

USER = os.environ['USER']    
PBS_PROJECT = 'NCGD0011'


def attrs_label(attrs): 
    """generate a label from long_name and units"""
    name = attrs["long_name"]
    units = '' if 'units' not in attrs else f' [{attrs["units"]}]' 
    return name + units


def get_ClusterClient(memory='25GB'):
    """get cluster and client"""
    cluster = PBSCluster(
        cores=1,
        memory=memory,
        processes=1,
        queue='casper',
        local_directory=f'/glade/scratch/{USER}/dask-workers',
        log_directory=f'/glade/scratch/{USER}/dask-workers',
        resource_spec=f'select=1:ncpus=1:mem={memory}',
        project=PBS_PROJECT,
        walltime='06:00:00',
        interface='ib0',
    )
    
    jupyterhub_server_name = os.environ.get('JUPYTERHUB_SERVER_NAME', None)    
    dashboard_link = 'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'    
    if jupyterhub_server_name:
        dashboard_link = (
            'https://jupyterhub.hpc.ucar.edu/stable/user/'
            + '{USER}'
            + f'/{jupyterhub_server_name}/proxy/'
            + '{port}/status'
        )
    dask.config.set({'distributed.dashboard.link': dashboard_link})        
    client = Client(cluster)
    return cluster, client


def retrieve_woa_dataset(variable, value):
    cat = intake.open_catalog("data/catalogs/woa2018-catalog.yml")
    if isinstance(variable, list):
        return xr.merge(
            [cat[v](time_code=value).to_dask() for v in variable]
        )
    else:
        return cat[variable](time_code=value).to_dask()

    
class timer(object):
    """support reporting timing info with named tasks"""
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.tic = time.time()
    
    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'[{self.name}]: ', end='')
        toc = time.time() - self.tic        
        print(f'{toc:0.5f}s')