"""
Tools for reading in solar structure data based on "model S", as reported in

     Christensen-Dalsgaard, J., Däppen, W., Ajukov, S. V., Anderson, E. R., Antia, H. M., Basu, S., Baturin, V. A., Berthomieu, G., Chaboyer, B., Chitre, S. M., Cox, A. N., Demarque, P., Donatowicz, J., Dziembowski, W. A., Gabriel, M., Gough, D. O., Guenther, D. B., Guzik, J. A., Harvey, J. W., Hill, F., Houdek, G., Iglesias, C. A., Kosovichev, A. G., Leibacher, J. W., Morel, P., Proffitt, C. R., Provost, J., Reiter, J., Rhodes Jr., E. J., Rogers, F. J., Roxburgh, I. W., Thompson, M. J., Ulrich, R. K., 1996.
     The current state of solar modeling.
     Science, 272, 1286 - 1292.

and as freely available at:

     https://users-phys.au.dk/~jcd/solar_models/

"""

import numpy as np
import pandas as pd

model_S_GONG = 'fgong.l5bi.d.15c'
# from: https://users-phys.au.dk/~jcd/solar_models/fgong.l5bi.d.15

model_S_limited = 'cptrho.l5bi.d.15c'
# from: https://users-phys.au.dk/~jcd/solar_models/cptrho.l5bi.d.15c

def read_model_S_GONG_format(file=model_S_GONG):
    """
    The GONG formatted data for model S includes a comprehensive set of variables, in a formatted ASCII file with fixed width columns.

    Details are given in:

         https://users-phys.au.dk/~jcd/solar_models/file-format.pdf,

    and the publically available files appear to be version 210 or earlier (25 data values before repeat).

    This routine extracts the header information, the grid data, and then returns both as a dictionary object containing pandas dataframes.
    """
    f = open(file, 'r')
    for i, line in enumerate(f):
        if i==4: # header line that contains: nn, iconst, ivar, ivers
            nn, iconst, ivar, ivers = list(map(int,line.split()))
    f.close()
    print('header info: {:d}, {:d}, {:d}, {:d}'.format(nn, iconst, ivar, ivers))

    # read in global model parameters
    global_parameters = pd.read_fwf(file, skiprows=4, nrows=3, widths=[16 for i in range(5)])
    # reshape and label; 15 variables are used
    col_names = ['M', 'R', 'L', 'Z', 'X0',
                 'α', 'φ', 'ξ', 'β', 'λ',
                 'd2lnP/dlnR2_c', 'd2lnρ/dlnR2_c', 'age', '14', '15']

    global_parameters = pd.DataFrame(global_parameters.values.reshape(1,iconst), columns=col_names)
    # read in stellar structure grid data
    data = pd.read_fwf(file, skiprows=7, widths=[16 for i in range(5)])
    # reshape and label; only the first 25 variables from the documentation are used
    col_names = ['r', 'ln q', 'T', 'p', 'ρ',
                 'X', 'L', 'κ', 'ε', 'Γ1',
                 'grad_ad', 'δ', 'c_P', 'μ_e_inv', 'entropy',
                 'rx', 'Z', 'R-r', 'ε_g', 'L_g',
                 'X3He', 'X12C', 'X13C', 'X14N', 'X16O']
    data = pd.DataFrame(data.values.reshape(len(data)//5, ivar), columns=col_names)
    structure = {'global':global_parameters, 'data':data}
    return structure

def read_model_S_limited_format(file=model_S_limited):
    """
    The limited formatted data for model S includes a smaller set of data in a whitespace delimited text file.

    This routine extracts the grid data, and then returns a dictionary object containing pandas dataframes.  The global data does not exist and is set to None.
    """
    col_names = ['r', 'c', 'rho', 'p', 'Gamma_1', 'T']
    data = pd.read_csv(file, delim_whitespace=True, comment='#', names=col_names)
    structure = {'global':None, 'data':data}
    return structure
