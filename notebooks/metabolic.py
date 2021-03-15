import numpy as np
import seawater as sw
from functools import partial

from scipy.optimize import newton

kb = 8.6e-5 # eV/K
kb_inv = 1./kb
R_gasconst = 8.3144621 # J/mol/K
T0_Kelvin = 273.15

Tref = 15. # °C
Tref_K = Tref + T0_Kelvin

XiO2 = 0.209 # Mean atmospheric O2 mixing ratio


def compute_ATmax(pO2, Ac, Eo, dEodT=0.):
    """
    Compute the maximum temperature at which resting or active (sustained)
    metabolic rate can be realized at a given po2.

    Parameters
    ----------
    Po2 : float
        Ambient O2 pressure (atm)

    Ac : float
        Hypoxia tolerance at Tref (1/atm) - can be either at rest (Ao) or at
        an active state.  For active thermal tolerance, argument should
        be Ac = Ao / Phi_crit

    Eo : float
        Temperature sensitivity of hypoxia tolerance (eV)
    
    dEdT: float
        Rate of change of Eo with temperature
    

    Note: Ac*Po2 must be unitless, but the units of either one are arbitrary

    Returns
    -------
    Tmax : float
        The 
    """
    
    def Phi_opt(T):
        return Phi(pO2, T, Ac, Eo, dEodT) - 1.
    
    # make a good initial guess for Tmax
    # - evaluate function over large temperature range
    # - find the zero crossings
    # - pick the highest 
    trange = np.arange(-2., 201., 1.)
    fvalue = Phi_opt(trange)
    fvalue[fvalue==0.] = np.nan
    sign = fvalue / np.abs(fvalue)
    ndx = np.where(sign[:-1] != sign[1:])[0]

    # no solution
    if len(ndx) == 0:
        return np.nan
    
    return newton(Phi_opt, trange[ndx[-1]])    


def Phi(pO2, T, Ac, Eo, dEodT=0.):
    """compute the metabolic index"""
    return Ac * pO2 * _Phi_exp(T, Eo, dEodT)


def pO2_at_Phi_crit(T, Ac, Eo, dEodT=0.):
    """compute pO2 at Φcrit"""
    return np.reciprocal(Ac * _Phi_exp(T, Eo, dEodT))


def _Phi_exp(T, Eo, dEodT):
    T_K = T + T0_Kelvin
    return np.exp(kb_inv * (Eo + dEodT * (T_K - Tref_K)) * (1./T_K - 1./Tref_K))


def compute_pO2(O2, T, S, depth):
    """
    Compute the partial pressure of O2 in seawater including 
    correction for the effect of hydrostatic pressure of the 
    water column based on Enns et al., J. Phys. Chem. 1964
      d(ln p)/dP = V/RT
    where p = partial pressure of O2, P = hydrostatic pressure
    V = partial molar volume of O2, R = gas constant, T = temperature
    
    Parameters
    ----------
    O2 : float
      Oxygen concentration (mmol/m3)
    
    T : float
       Temperature (°C)
    
    S : float
       Salinity 
       
    depth : float
       Depth (m)
       
    Returns
    -------
    pO2 : float
       Partial pressure (atm)
       
    """
    
    V = 32e-6 # partial molar volume of O2 (m3/mol)
    Patm = 1. # Atm pressure
    
    T_K = T + T0_Kelvin
    
    db2Pa = 1e4 # convert pressure: decibar to Pascal

    # Solubility with pressure effect
    P = sw.pres(depth, lat=0.)  # seawater pressure [db] !! Warning - z*0 neglects gravity differences w/ latitude
    rho = sw.dens(S, T, depth)  # seawater density [kg/m3]

    dP = P * db2Pa
    pCor = np.exp(V * dP / (R_gasconst * T_K))

    Kh = 1e-3 * O2sol(S, T) * rho / (Patm * XiO2) # solubility [mmol/m3/atm]
    
    return (O2 / Kh) * pCor


def O2sol(S, T):
    """
    Solubility of O2 in sea water
    INPUT:
    S = salinity    [PSS]
    T = temperature [degree C]

    conc = solubility of O2 [mmol/m^3]

    REFERENCE:
    Hernan E. Garcia and Louis I. Gordon, 1992.
    "Oxygen solubility in seawater: Better fitting equations"
    Limnology and Oceanography, 37, pp. 1307-1312.
    """

    # constants from Table 4 of Hamme and Emerson 2004
    return _garcia_gordon_polynomial(S, T,
                                     A0 = 5.80871,
                                     A1 = 3.20291,
                                     A2 = 4.17887,
                                     A3 = 5.10006,
                                     A4 = -9.86643e-2,
                                     A5 = 3.80369,
                                     B0 = -7.01577e-3,
                                     B1 = -7.70028e-3,
                                     B2 = -1.13864e-2,
                                     B3 = -9.51519e-3,
                                     C0 = -2.75915e-7)


def _garcia_gordon_polynomial(S,T,
                              A0 = 0., A1 = 0., A2 = 0., A3 = 0., A4 = 0., A5 = 0.,
                              B0 = 0., B1 = 0., B2 = 0., B3 = 0.,
                              C0 = 0.):

    T_scaled = np.log((298.15 - T) /(T0_Kelvin + T))
    return np.exp(A0 + A1*T_scaled + A2*T_scaled**2. + A3*T_scaled**3. + A4*T_scaled**4. + A5*T_scaled**5. + \
                  S*(B0 + B1*T_scaled + B2*T_scaled**2. + B3*T_scaled**3.) + C0 * S**2.)


