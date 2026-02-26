# -*- coding: utf-8 -*-


### Calculating Tb from Sigma_H2 & Tex
###  assuming LTE condition and optically thin


# import modules
import os
import sys
import math
import glob
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy import constants, units


# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)

# path to here
path_to_here = os.path.dirname(__file__)
path_to_library = path_to_here[:-11]
#print (path_to_here)
#print (path_to_library)


class LTEAnalysis():

    def __init__(self, molecules):
        '''
        Python class for Local Thermal Equilibrium (LTE) analysis.

        Args
        ----
        molecules (str or list or tuple): molecule(s) to be analized.
        '''
        self.moldata = {}

        if (type(molecules) == list) or (type(molecules) == tuple):
            for mol in molecules:
                self.read_lamda_moldata(mol)
        elif type(molecules) == str:
            self.read_lamda_moldata(molecules)
        else:
            print('ERROR\tLTEAnalysis: molecules must be str, or list or tuple of strings.')


    # read molecular data
    def read_lamda_moldata(self, line):
        '''
        Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database)
        '''
        # find path
        line = line.lower()
        if '+' in line: line = line.replace('+','p')
        infile = glob.glob(path_to_library+'moldata/'+line+'.dat')
        if len(infile) == 0:
            print('ERROR\tread_lamda_moldata: Cannot find LAMDA file.')
            print('ERROR\tread_lamda_moldata: Only C18O, CO, and N2H+ are \
                supported for now')
            return
        else:
            data = pd.read_csv(infile[0], comment='!', 
                delimiter='\r\n', header=None, engine='python')

        # get
        # line name, weight, nlevels
        _, weight, nlevels = data[0:3][0].values
        weight  = float(weight)
        nlevels = int(nlevels)

        # energy on each excitation level
        elevels = data[3:3+nlevels].values
        elevels = np.array([ elevels[i][0].split() for i in range(nlevels)])
        lev, EJ, gJ, J = elevels.T
        lev = np.array([ int(lev[i]) for i in range(nlevels)])
        EJ  = np.array([ float(EJ[i]) for i in range(nlevels)]) \
        * clight * hp / kb # in K
        gJ  = np.array([ float(gJ[i]) for i in range(nlevels)])
        J   = np.array([ int(J[i]) for i in range(nlevels)])

        # number of transition
        ntrans = data[0][3+nlevels].strip()
        ntrans = int(ntrans)

        # Einstein A coefficient
        vtrans = data[3+nlevels+1:3+nlevels+1+ntrans].values
        vtrans = np.array([vtrans[i][0].split() for i in range(ntrans)])

        itrans, Jup, Jlow, Acoeff, freq, Eu = vtrans.T
        itrans = np.array([ int(itrans[i]) for i in range(ntrans)])
        Jup    = np.array([ int(Jup[i]) for i in range(ntrans)])
        Jlow   = np.array([ int(Jlow[i]) for i in range(ntrans)])
        Acoeff = np.array([ float(Acoeff[i]) for i in range(ntrans)])
        freq   = np.array([ float(freq[i]) for i in range(ntrans)])
        Eu   = np.array([ float(Eu[i]) for i in range(ntrans)])
        #for i in range(ntrans):
        #    print('Eu, Eu: %.4f %.4f'%(EJ[i+1], delE[i]))

        # transitions
        trans = [ str(J[ int(Jup[i] - 1)]) + '-' \
        + str( J[ int(Jlow[i] - 1)]) for i in range(len(itrans))]

        # save
        self.moldata[line] = {
        'weight':weight,
        'nlevels': nlevels,
        'EJ': EJ,
        'gJ': gJ,
        'J': J,
        'ntrans': ntrans,
        'trans': trans,
        'Jup': Jup,
        'Jlow': Jlow,
        'Acoeff': Acoeff,
        'freq': freq,
        'Eu': Eu,
        }

        #return line, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE


    def to_tau(self, Ntot, Tex, delv):
        '''
        Calculate tau from given the total column density, excited temeprature 
         and linewidth.

        Parameters
        ----------
         Ntot (array or float): The total number column density of the molecule (cm^-2).
         Tex (array or float): The excitation temperature (K).
         delv (array or float): The linewidth (cm/s).
        '''
        # line Ju --> Jl
        freq_ul = self.moldata[line]['freq'][Ju-1] * 1e9 # Hz
        Aul     = self.moldata[line]['Acoeff'][Ju-1]
        gu      = self.moldata[line]['gJ'][Ju]
        gl      = self.moldata[line]['gJ'][Ju-1]
        Eu     = self.moldata[line]['EJ'][Ju]
        El     = self.moldata[line]['EJ'][Ju-1]
        #EJu     = self.moldata[line]['Eu'][Ju - 1]
        #print(Ju, EJu)

        # partition function
        Qrot = Pfunc(self.moldata[line]['EJ'], self.moldata[line]['gJ'], 
            self.moldata[line]['J'], Tex)

        return (clight*clight*clight)/(8.*np.pi*freq_ul*freq_ul*freq_ul)*(gu/Qrot)\
        *np.exp(-Eu/Tex)*Ntot*Aul*(np.exp(hp*freq_ul/(kb*Tex)) - 1.) / delv


    def get_intensity(self, line, Ju, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau=False):
        '''
        Calculate the intensity or brightness temperature of a molecular line transition.
        Currently only LTE assumption is supported. Non-LTE calculation using RADEX will be 
        implemented in future.

        Parameter
        ---------
                line (str): Name of line.
                Ju (int): Upper excitation level.
                Tex (float): Excitation temperature (K).
                Ncol (float): Number column density of H2 or the molecule for which the intensity
                 is calculated (cm^-2).
                delv (float): FWHM of the line profile (cm s^-1).
                mode (str): LTE or non-LTE. Currently only LTE assumption is supported.
                Xconv (float): Conversion factor from the H2 column density to the column density of the
                 target molecule. If this is not given, then the input column density is regarded as the 
                 column density of the target molecule.
                Tbg (float): Temperature of the background emission (K).
                Tb: (bool): If True, the output will be in the brightness temperature. Otherwise, the output
                 will be in intensity in a unit of cgs.
                return_tau (bool): If True, the optical depth tau will be returned instead of the intensity.
        '''
        # line
        if line in self.moldata.keys():
            pass
        else:
            self.read_lamda_moldata(line)

        # line Ju --> Jl
        freq_ul = self.moldata[line]['freq'][Ju-1] * 1e9 # Hz
        Aul     = self.moldata[line]['Acoeff'][Ju-1]
        gu      = self.moldata[line]['gJ'][Ju]
        gl      = self.moldata[line]['gJ'][Ju-1]
        Eu     = self.moldata[line]['EJ'][Ju]
        El     = self.moldata[line]['EJ'][Ju-1]
        #EJu     = self.moldata[line]['Eu'][Ju - 1]
        #print(Ju, EJu)

        # partition function
        Qrot = Pfunc(self.moldata[line]['EJ'], self.moldata[line]['gJ'], 
            self.moldata[line]['J'], Tex)

        # N_H2 --> N_mol
        if Xconv is not None: Ncol *= Xconv

        # tau_v
        _delv = delv if lineprof == 'rect' else delv * 0.5 * np.sqrt(np.pi / np.log(2.))
        tau_v = (clight*clight*clight)/(8.*np.pi*freq_ul*freq_ul*freq_ul)*(gu/Qrot)\
        *np.exp(-Eu/Tex)*Ncol*Aul*(np.exp(hp*freq_ul/(kb*Tex)) - 1.)/_delv
        #print('tau = %.2e'%tau_v)

        if return_tau:
            return tau_v
        #Iv = Bv(Tbg,freq_ul)*np.exp(-tau_v) + Bv(Tex, freq_ul)*(1. - np.exp(-tau_v))
        #Iv -= Bv(Tbg,freq_ul)
        Iv = (Bv(Tex, freq_ul) - Bv(Tbg,freq_ul)) * (1. - np.exp(-tau_v))

        if Tb:
            # equivalent brightness temperature
            #return (Tex - Tbg) * (1. - np.exp(-tau_v))
            return (clight*clight/(2.*freq_ul*freq_ul*kb))*Iv
        else:
            return Iv


    def get_tbratio(self, line, Ju_u, Ju_l, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73):
        '''
        Get Tb ratio between two transitions.
        '''

        tb_u = self.get_intensity(line, Ju_u, Tex, Ncol, delv, lineprof=lineprof, 
            mode=mode, Xconv=Xconv, Tbg=Tbg, return_tau=False, Tb=True)
        tb_l = self.get_intensity(line, Ju_l, Tex, Ncol, delv, lineprof=lineprof, 
            mode=mode, Xconv=Xconv, Tbg=Tbg, return_tau=False, Tb=True)

        return tb_u / tb_l


    def get_ltemass(self, line, Fv, Ju, Tex, Xconv,
        dist=140., mu=2.8, Tbg = 2.73, 
        S_TA=None, bmaj=None, bmin=None, funit = 'Jy',
        dx = None, dy = None):
        '''
        Calculate LTE mass

        Parameters
        ----------
         line (str): Name of the molecule.
        '''
        # line
        if line in self.moldata.keys():
            pass
        else:
            self.read_lamda_moldata(line)

        # line Ju --> Jl
        freq_ul = self.moldata[line]['freq'][Ju-1]*1e9 # Hz
        Aul     = self.moldata[line]['Acoeff'][Ju-1]
        gu      = self.moldata[line]['gJ'][Ju]
        gl      = self.moldata[line]['gJ'][Ju-1]
        EJu     = self.moldata[line]['EJ'][Ju]
        #print(Aul, freq_ul, Ju, gu, gl)

        # partition function
        Qrot = Pfunc(self.moldata[line]['EJ'], self.moldata[line]['gJ'], 
            self.moldata[line]['J'], Tex)
        #print('Qrot %.2f'%Qrot)
        print('Tex %.2f K'%Tex)
        #Qrot = 5.7

        # coefficients
        #C1 = Qrot * (4. * np.pi * mu * mp)/(hp * clight * gu * Aul)
        # more precisely
        dJv = Bv(Tex, freq_ul) - Bv(Tbg, freq_ul) # cgs
        c1        = 8. * np.pi * freq_ul**3. * Qrot
        c2        = clight * clight * clight * gu * Aul
        exp       = np.exp(EJu/Tex)
        exp2      = np.exp(hp * freq_ul / (kb * Tex)) - 1.

        # observed flux
        #if S_TA:
        if (funit == 'K') | (funit == 'Kelvin'):
            '''
            if bmaj == None:
                print ('ERROR\tLTEmass: bmaj and bmin must be given\
                 for conversion from K arcsec2 km/s --> Jy km/s.')
                return
            elif bmin == None:
                print ('ERROR\tLTEmass: bmaj and bmin must be given\
                 for conversion from K arcsec2 km/s --> Jy km/s.')
                return

            # convert units
            #bmaj = bmaj*np.pi/(180.*60.*60.) # arcsec --> radian
            #bmin = bmin*np.pi/(180.*60.*60.) # arcsec --> radian
            Fv       = Fv*S_TA                         # K arcsec2 km/s --> Jy/beam arcesc2 km/s
            beamsize = np.pi/(4.*np.log(2.))*bmaj*bmin # beam --> arcsec2
            Fv       = Fv/beamsize                     # --> Jy km/s
            '''
            #if (dx is None) | (dy is None):
            #    print ('ERROR\tLTEmass: dx and dy must be given\
            #     for conversion from K arcsec2 --> to flux.')
            #    return
            dJv *= clight * clight * 0.5 / freq_ul / freq_ul / kb # intensity to equivalent brightness temperature
            dA_D = dist * dist * au * au # arcsec^2 --> cm^2
            Fv *= 1.e5 # km/s --> cm/s
        else:
            dA_D = dist * dist * pc * pc # cm^2
            # From Jy km/s to cgs
            Fv = Fv*1.0e-26         # Jy km/s --> MKS (1 Jy = 10^-26 Wm-2Hz-1)
            Fv = Fv*1.e7*1.e5*1.e-4 # mks --> cgs (erg cm^-2 cm/s)

        # Mgas (assuming optically thin)
        #Mgas = C1 * np.exp(EJu/Tex) * dist_pc * dist_pc * Fv/Xconv # g
        Mgas = Fv * c1 / c2 * exp / exp2 / dJv * dA_D * mu * mp / Xconv
        Mgas = Mgas / ms # g --> Msun

        print ('Mgas: %.4f Msun'%Mgas)

        return Mgas


    def get_column(self, line, Iint, bmaj, bmin, 
        Ju, Tex, Xconv, dist=140., 
        tau=False, delv=None, sigma=False, Tbg=2.73, number=True,
        err_Ivint=0., err_Tex=0., err_delv=0.):
        '''
        Calculate gas column density from line

        Args:
          Iving: integrated intensity [Jy/beam km/s]
          Tk: kinetic temperature [K]
          bmaj, bmin: beam size along beam major and minor axis [arcsec]
          Ju(int): upper excitation state
          Z: partision function
          Xconv: conversion factor
          dist: distance to the object [pc]
          Tbg: background temperature [K]

          (parameters of molecules)
          Acoeff: Einstein A coefficient [s^-1]
          freq: frequency [GHz]
          delE: transitional energy [K]
          EJ: energy at J [cm^-1]
          gJ: statistical weight
          J: energy level
          mu: molecular weight
          Brot: rotational constant [s^-1]

        Return:
          Sigma_H2O [g cm^-2]
        '''

        # line
        if line in self.moldata.keys():
            pass
        else:
            self.read_lamda_moldata(line)

        # line Ju --> Jl
        freq_ul = self.moldata[line]['freq'][Ju-1]*1e9 # Hz
        Aul     = self.moldata[line]['Acoeff'][Ju-1]
        gu      = self.moldata[line]['gJ'][Ju]
        gl      = self.moldata[line]['gJ'][Ju-1]
        EJu     = self.moldata[line]['EJ'][Ju]
        mu      = self.moldata[line]['weight']

        # partition function
        Qrot = Pfunc(self.moldata[line]['EJ'], self.moldata[line]['gJ'], 
            self.moldata[line]['J'], Tex)

        # !!! start !!!
        dist_pc = dist * pc # pc --> cm
        #C1 = Qrot * (4. * np.pi * mu * mp)/(hp * clight * gu * Aul)

        # convert units
        bmaj = bmaj*np.pi/(180.*60.*60.) # arcsec --> radian
        bmin = bmin*np.pi/(180.*60.*60.) # arcsec --> radian


        if tau:
            # derive column density from optical depth tau
            if delv:
                pass
            else:
                print ('ERROR: delv is necessary if tau=True')
                return

            delv = delv * 1.e5 # km/s --> cm/s

            if sigma:
                term_delv = np.sqrt(2.*np.pi)*delv
            else:
                term_delv = np.sqrt(np.pi)*delv/(2.*np.sqrt(np.log(2.)))

            tau_tot   = Ivint
            c1        = 8. * np.pi * freq_ul**3. * Qrot
            c2        = clight * clight * clight * gu * Aul
            exp       = np.exp(EJu/Tex)
            exp2      = np.exp(hp * freq_ul / (kb * Tex)) - 1.
            N_mol     = tau_tot * term_delv * c1 / c2 * exp / exp2
            Sigma_mol = N_mol * mu * mp

            # Calculate the error propagation.
            dN_dtau  = N_mol/tau_tot
            dN_ddelv = N_mol/delv
            dN_dTex  = c1 / c2 * tau_tot * term_delv * Tex**(-2.) * exp * exp2**(-2.)\
             *((-EJu + hp * freq_ul / kb) * np.exp(hp * freq_ul / (kb * Tex)) + EJu)

            err_Nmol   = np.sqrt((dN_dtau*err_Ivint)**2. 
                + (dN_ddelv*err_delv)**2. 
                + (dN_dTex*err_Tex)**2. )
            err_Sigmol = err_Nmol * mu * mp
        else:
            # derive column density from the integrated intensity

            # Jy/beam -> Jy/str
            # Omg_beam (str) = (pi/4ln(2))*beam (rad^2)
            # I [Jy/beam] / Omg_beam = I [Jy/str]
            # beam area = Omega_beam*d^2
            C2     = np.pi / (4.*np.log(2.)) # beam(rad) -> beam (sr)
            bTOstr = bmaj * bmin * C2          # beam --> str

            Istr = Iint/bTOstr          # Jy/beam km/s --> Jy/str km/s
            Istr = Istr*1.0e-26          # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
            Istr = Istr*1.e7*1.e-4*1.e5  # MKS --> cgs (erg s^-1 cm^-2 Hz^-1 str^-1 cm/s)
            err_Istr = err_Ivint/bTOstr
            err_Istr = err_Istr*1.0e-26*1.e7*1.e-4*1.e5

            # coefficients
            c1  = 8. * np.pi * freq_ul**3. * Qrot
            c2  = clight * clight * clight * gu * Aul
            exp = np.exp(EJu/Tex)
            jterm = Bv(Tex, freq_ul) - Bv(Tbg, freq_ul)
            exp2  = np.exp(hp * freq_ul / (kb * Tex)) - 1.

            N_mol     = Istr*c1/c2*exp/exp2/jterm # cm^-2
            Sigma_mol = N_mol * mu * mp           # g cm^-2

            # Calculate the error propagation.
            dN_dIvint  = N_mol/Istr
            err_Nmol   = np.sqrt((dN_dIvint*err_Istr)**2. )
            err_Sigmol = err_Nmol * mu * mp

        # column density of H2O gas
        N_H2      = N_mol/Xconv
        Sigma_H2  = Sigma_mol/Xconv

        if number:
            print ('N_%s: %4e cm^-2'%(line, N_mol))
            print ('Uncertainty: %4e cm^-2'%err_Nmol)
            #print ('N_H2: %4e cm^-2'%N_H2)
            return N_mol #, N_H2
        else:
            print ('Sigma_%s: %4e g cm^-2'%(line, Sigma_mol))
            print ('Uncertainty: %4e g cm^-2'%err_Sigmol)
            #print ('Sigma_H2: %4e g cm^-2'%Sigma_H2)
            return Sigma_mol #, Sigma_H2



    def makegrid(self, lines, J1, J2, Texes, Ncols, delv, lineprof='rect', 
        mode='lte', Xconv=[], Tbg=2.73, Tb=True, fig=None, ax=None, lw=1., aspect=1.,
        xtype = 'int', ytype = 'int'):
        '''
        Produce a grid for the line intensity ratio for two transitions.

        Parameters
        ----------
            lines (str or list): Name of lines. A single line can be given as the str object.
             Two different lines can be given as a list of two line names.
            J1 (int): Upper excitation level of the first transition.
            J2 (int): Upper excitation level of the second transition.
            Texes (array): Array of excitation temperatures (K), with which the grid will be calculated.
            Ncols (array): Array of H2 or molecule's number column densities, with which 
             the grid will be calculated (cm^-2).
            delv (float): FWHM of the line profile (cm s^-1).
            mode (str): LTE or non-LTE. Currently only LTE assumption is supported.
            Xconv (float or list): Conversion factors from the H2 column density to the column density of the
             target molecule. Can be given as float for a single molecule, or as a list for two different kinds.
             If this is not given, then the input column density is regarded as the 
             column density of the target molecule.
            Tbg (float): Temperature of the background emission (K).
            Tb: (bool): If True, the output will be in the brightness temperature. Otherwise, the output
             will be in intensity in a unit of cgs.
        '''
        # Parameters
        if type(lines) is str:
            lines = [lines]*2
        elif type(lines) is list:
            if len(lines) !=2:
                print("ERROR\tget_grid: More than two elements are given for 'lines'.")
                print("ERROR\tget_grid: lines must be str or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'lines' is wrong.")
            print("ERROR\tget_grid: lines must be str or list with two elements.")
            return 0

        if type(Xconv) is float:
            Xconv = [Xconv]*2
        elif type(Xconv) is list:
            if len(Xconv) == 0:
                Xconv = [None]*2
            elif len(Xconv) != 2:
                print("ERROR\tget_grid: More than two elements are given for 'Xconv'.")
                print("ERROR\tget_grid: Xconv must be float or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'Xconv' is wrong.")
            print("ERROR\tget_grid: Xconv must be float or list with two elements.")
            return 0

        
        # define instant functions
        if xtype == 'int':
            f_tb1 = lambda Tex, Ncol: self.get_intensity(
                lines[0], J1, Tex, Ncol, delv, 
                lineprof=lineprof, mode=mode, Xconv=Xconv[0], 
                Tbg=Tbg, return_tau=False, Tb=Tb)
            xlabel = r'$T_\mathrm{b}$ (%s)'%(self.moldata[lines[0]]['trans'][J1 - 1])
        elif xtype == 'ratio':
            f_tb1 = lambda Tex, Ncol: self.get_tbratio(
                lines[0], J1[0], J1[1], Tex, Ncol, delv, 
                lineprof=lineprof, mode=mode, Xconv=Xconv[0], Tbg=Tbg,)
            xlabel = r'$T_\mathrm{b}$ (%s)$/T_\mathrm{b}$ (%s)'%(
                self.moldata[lines[0]]['trans'][J1[0] - 1], 
                self.moldata[lines[0]]['trans'][J1[1] - 1])
        else:
            print('ERROR\tmakegrid: ytype must be int or ratio.')
            return 0
        
        if ytype == 'int':
            f_tb2 = lambda Tex, Ncol: self.get_intensity(
                lines[1], J2, Tex, Ncol, delv, 
                lineprof=lineprof, mode=mode, Xconv=Xconv[1], 
                Tbg=Tbg, return_tau=False, Tb=Tb)
            ylabel = r'$T_\mathrm{b}$(%s)'%(self.moldata[lines[1]]['trans'][J2 - 1])
        elif ytype == 'ratio':
            f_tb2 = lambda Tex, Ncol: self.get_tbratio(
                lines[1], J2[0], J2[1], Tex, Ncol, delv, 
                lineprof=lineprof, mode=mode, Xconv=Xconv[1], Tbg=Tbg,)
            ylabel = r'$T_\mathrm{b}$(%s) $/$ $T_\mathrm{b}$(%s)'%(
                self.moldata[lines[1]]['trans'][J2[0] - 1], 
                self.moldata[lines[1]]['trans'][J2[1] - 1])
        else:
            print('ERROR\tmakegrid: ytype must be int or ratio.')
            return 0

        # figure
        if (fig is None) and (ax is None):
            fig = plt.figure()#figsize=(11.69, 8.27))
            ax  = fig.add_subplot(111)
        elif (fig is not None) and (ax is None):
            ax  = fig.add_subplot(111)

        # iso-temperature curves
        for i, Tex_i in enumerate(Texes):
            tb1 = []
            tb2 = []
            for j, Ncol_i in enumerate(np.logspace(np.log10(Ncols[0]), np.log10(Ncols[-1]),128)):
                tb1.append(f_tb1(Tex_i, Ncol_i))
                tb2.append(f_tb2(Tex_i, Ncol_i))

            ax.plot(tb1, tb2, c=cm.coolwarm(float(i+1)/len(Texes)), lw=lw)


        # iso-density curves
        for i, Ncol_i in enumerate(Ncols):
            tb1 = []
            tb2 = []
            for j, Tex_i in enumerate(np.linspace(Texes[0], Texes[-1],128)):
                tb1.append(f_tb1(Tex_i, Ncol_i))
                tb2.append(f_tb2(Tex_i, Ncol_i))
            ax.plot(tb1, tb2, c='k', lw=lw)#)cm.BrBG(float(i+0.5)/len(Ncols)))


        # aspect ratio
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        change_aspect_ratio(ax, aspect)

        return fig, ax


    '''
    def makegrid_old(self, lines, J1, J2, Texes, Ncols, delv, lineprof='rect', 
        mode='lte', Xconv=[], Tbg=2.73, Tb=True, fig=None, ax=None, lw=1., aspect=1.):
        Produce a grid for the line intensity ratio for two transitions.

        Parameters
        ----------
            lines (str or list): Name of lines. A single line can be given as the str object.
             Two different lines can be given as a list of two line names.
            J1 (int): Upper excitation level of the first transition.
            J2 (int): Upper excitation level of the second transition.
            Texes (array): Array of excitation temperatures (K), with which the grid will be calculated.
            Ncols (array): Array of H2 or molecule's number column densities, with which 
             the grid will be calculated (cm^-2).
            delv (float): FWHM of the line profile (cm s^-1).
            mode (str): LTE or non-LTE. Currently only LTE assumption is supported.
            Xconv (float or list): Conversion factors from the H2 column density to the column density of the
             target molecule. Can be given as float for a single molecule, or as a list for two different kinds.
             If this is not given, then the input column density is regarded as the 
             column density of the target molecule.
            Tbg (float): Temperature of the background emission (K).
            Tb: (bool): If True, the output will be in the brightness temperature. Otherwise, the output
             will be in intensity in a unit of cgs.
        # Parameters
        if type(lines) is str:
            lines = [lines]*2
        elif type(lines) is list:
            if len(lines) !=2:
                print("ERROR\tget_grid: More than two elements are given for 'lines'.")
                print("ERROR\tget_grid: lines must be str or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'lines' is wrong.")
            print("ERROR\tget_grid: lines must be str or list with two elements.")
            return 0

        if type(Xconv) is float:
            Xconv = [Xconv]*2
        elif type(Xconv) is list:
            if len(Xconv) == 0:
                Xconv = [None]*2
            elif len(Xconv) != 2:
                print("ERROR\tget_grid: More than two elements are given for 'Xconv'.")
                print("ERROR\tget_grid: Xconv must be float or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'Xconv' is wrong.")
            print("ERROR\tget_grid: Xconv must be float or list with two elements.")
            return 0

        # figure
        if (fig is None) and (ax is None):
            fig = plt.figure()#figsize=(11.69, 8.27))
            ax  = fig.add_subplot(111)
        elif (fig is not None) and (ax is None):
            ax  = fig.add_subplot(111)

        # iso-temperature curves
        for i, Tex_i in enumerate(Texes):
            tb1 = []
            tb2 = []
            for j, Ncol_i in enumerate(np.logspace(np.log10(Ncols[0]), np.log10(Ncols[-1]),128)):
                #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
                tb1.append(self.get_intensity(lines[0], J1, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[0], Tbg=Tbg, return_tau=False, Tb=Tb))
                tb2.append(self.get_intensity(lines[1], J2, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[1], Tbg=Tbg, return_tau=False, Tb=Tb))

            ax.plot(tb1, tb2, c=cm.coolwarm(float(i+1)/len(Texes)), lw=lw)


        # iso-density curves
        for i, Ncol_i in enumerate(Ncols):
            tb1 = []
            tb2 = []
            for j, Tex_i in enumerate(np.linspace(Texes[0], Texes[-1],128)):
                #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
                tb1.append(self.get_intensity(lines[0], J1, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[0], Tbg=Tbg, return_tau=False, Tb=Tb))
                tb2.append(self.get_intensity(lines[1], J2, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[1], Tbg=Tbg, return_tau=False, Tb=Tb))

            ax.plot(tb1, tb2, c='k', lw=lw)#)cm.BrBG(float(i+0.5)/len(Ncols)))


        # aspect ratio
        ax.set_xlabel(r'$T_\mathrm{b}$(%i-%i)'%(J1, J1-1))
        ax.set_ylabel(r'$T_\mathrm{b}$(%i-%i)'%(J2, J2-1))
        change_aspect_ratio(ax, aspect)

        return fig, ax
    '''



    def solve_ntrans(self, mol, Ju, 
        Tb, e_Tb, delv, p0, lineprof='gauss', 
        e_flux = 0., Xconv=None, Tbg=2.73,
        nmc = 3000, outname = None, savefig = False,
        showfig = False, Tex_plt = np.arange(10., 60., 10.),
        Ncol_plt = np.logspace(9,12,6)):
        '''
        Solve LTE equation to get Tex and Ncol for given Tb of multiple transition lines for a molecule.

        Parameters
        ----------
         mol (str): molecule for which the LTE equation is solved.
         Ju_list (list): J of the upper energy state.
         Tb (array): Tb array in a shape of (n, m), where n is the number of rotational
          transitions and m is the number of data points.
         e_Tb (array): Uncertainties of Tb.
         delv (float): Line width.
        '''
        # for plot
        import corner
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import arviz as az

        def n2diagram(ax, Tb, e_Tb, popt):
            # make grid
            _, ax = self.makegrid([mol, mol], Ju[0], Ju[1], 
                Tex_plt, Ncol_plt, delv, lineprof=lineprof, 
                Xconv=[Xconv, Xconv], Tbg=2.73, Tb=True, lw=1., aspect=1.,
                ax = ax)
            # data
            ax.errorbar(Tb[0], Tb[1], xerr= e_Tb[0], yerr = e_Tb[1],
                ls = '', capsize = 2., capthick = 1., color = 'k'
                )
            # solution
            Tb_fit = np.array([func(Ju[i],*popt) for i in range(len(Ju))])
            ax.scatter(Tb_fit[0], Tb_fit[1], color = 'r', alpha = 0.5)


        # number of transitions
        ntrans = len(Ju)
        if type(Ju) == list:
            Ju = np.array(Ju)

        # fitting function
        func = lambda Ju, Tex, Ncol: self.get_intensity(mol, Ju, Tex, Ncol, delv, 
            lineprof=lineprof, Xconv=Xconv, Tbg=Tbg, Tb=True, return_tau=False)


        # fitting
        if len(Tb.shape) == 1:
            # fit
            pfit, popt, perr = mcsolver(func, p0, Ju, Tb, e_Tb, e_flux, nmc = nmc)
            # plot
            if any([savefig, showfig]):
                fig = corner.corner(pfit.T, 
                    labels = [r'$T_\mathrm{ex}$', r'$N_\mathrm{col}$'],
                    range = [0.95]*2)
                # subplot
                axes = fig.get_axes()
                ax = inset_axes(axes[1], width = '70%', height = '70%',
                    loc = 'upper right')
                n2diagram(ax, Tb, e_Tb, popt)

                if savefig: fig.savefig(outname + '.pdf', transparent = True)
                if showfig: plt.show()
                plt.close()
            return popt, perr
        elif len(Tb.shape) == 2:
            # shape
            n, m = Tb.shape
            if n != ntrans:
                Tb = Tb.T
                e_Tb = e_Tb.T
                n, m = Tb.shape
            if n > 2:
                print('CAUTION\tsolve_ntrans: currently no plot option is available for n > 2.')
                savefig = False
                showfig = False

            # output array
            popt_out = np.empty((2, m))
            perr_out = np.empty((2, 2, m))

            # fitting
            for i in range(m):
                # fit
                pfit, popt, perr = mcsolver(func, p0, Ju, Tb[:,i], e_Tb[:,i], 
                    e_flux, nmc = nmc)
                popt_out[:,i] = popt
                perr_out[:,:,i] = perr

                # plot
                if any([savefig, showfig]):
                    fig = corner.corner(pfit.T, labels = [r'$T_\mathrm{ex}$', r'$N_\mathrm{col}$'],
                        range = [0.95]*2)
                    # subplot
                    axes = fig.get_axes()
                    ax = inset_axes(axes[1], width = '70%', height = '70%',
                        loc = 'upper right')
                    n2diagram(ax, Tb[:,i], e_Tb[:,i], popt)

                    if savefig: fig.savefig(outname + '_m%i'%i + '.pdf', transparent = True)
                    if showfig: plt.show()
                    plt.close()
            return popt_out, perr_out
        else:
            print('ERROR\tsolve_ntrans: input Tb must be in one or two dimension.')
            return 0



# partition function
def Pfunc(EJ, gJ, J, Tex):
    '''
    Calculate the partition function.

    Args:
        EJ: energy at energy level J
        gJ: statistical weight
        J: energy level
        Tk: kinetic energy

    Return:
        Z: partition function
    '''
    Zarray = np.array([gJ[j]*np.exp(-EJ[j]/Tex) for j in range(len(J))])
    return np.sum(Zarray)


# planck function
def Bv(T,v):
    '''
    Planck function

    Args:
        T: temprature [K]
        v: frequency [Hz]
    '''
    exp   = np.exp((hp*v)/(kb*T)) - 1.0
    fterm = (2.0 * hp * v * v * v)/(clight * clight)
    return fterm / exp



def change_aspect_ratio(ax, ratio, plottype='linear'):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    if plottype == 'linear':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    elif plottype == 'loglog':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    elif plottype == 'linearlog':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
    elif plottype == 'loglinear':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    else:
        print('ERROR\tchange_aspect_ratio: plottype must be choosen from the types below.')
        print('   plottype can be linear or loglog.')
        print('   plottype=loglinear and linearlog is being developed.')
        return

    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)


def mcsolver(func, p0, x, y, e_y, e_f = 0., nmc = 3000,
    simple_out = False, credible_interval = 0.68):
    # module
    from tqdm import tqdm
    from scipy import optimize, stats
    import arviz as az

    # chi to be minimized
    def chi(p, x, y, e_y,):
        return (y - func(x, *p)) / e_y

    # random sampling
    pfit = np.empty((len(p0), nmc))
    ysmpl = np.array([
        np.random.normal(y[i], e_y[i], nmc) for i in range(len(y))
        ])
    scale = np.random.normal(1., e_f, nmc) if e_f > 0. else np.ones(nmc)

    # fitting
    pfit = np.array([
        optimize.leastsq(chi, p0, 
            args=(x, ysmpl[:,i] * scale[i], e_y * scale[i]), full_output=True)[0]
        for i in tqdm(range(nmc))
        ]).T

    # output
    if simple_out:
        popt = np.nanmean(pfit, axis = 1)
        #popt, _ = stats.mode(pfit, axis = 1, nan_policy='omit')
        perr = np.sqrt(np.nanvar(pfit, axis = 1))
    else:
        pstat = np.percentile(pfit, 
                [50*(1. - credible_interval), 50, 50*(1. + credible_interval)],
                axis = 1)
        #print(pstat)
        q = np.diff(pstat, axis=0)
        popt = pstat[1,:]
        perr = q[:, :]

    return pfit, popt, perr