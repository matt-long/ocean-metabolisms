import os

import numpy as np
import xarray as xr

import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

USER = os.environ['USER']    
PBS_PROJECT = 'UGIT0016'


def attrs_label(attrs): 
    """generate a label from long_name and units"""
    return f'{attrs["long_name"]} [{attrs["units"]}]'


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


