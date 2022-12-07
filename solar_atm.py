import numpy as np
import scipy as scp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
        plt.style.use('ggplot')
except:
        print("Upgrade matplotlib; for now we're falling back to old plot styles")

import dedalus.public as de


def plot_abs(x, y, ax, color=None, color_pos=None, color_neg=None, **kwargs):
    pos_mask = np.logical_not(y>0)
    neg_mask = np.logical_not(y<0)
    pos_line = np.ma.MaskedArray(y, pos_mask)
    neg_line = np.ma.MaskedArray(y, neg_mask)

    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']

    if color_pos is None:
        color_pos = color

    if color_neg is None:
        color_neg = color
        
    ax.plot(x, pos_line, color=color_pos, **kwargs)
    ax.plot(x, np.abs(neg_line), color=color_neg, linestyle='dashed')
    
def read_atm(filename="solar_atm.txt"):
    data = np.genfromtxt(filename)
    data_dict = {}
    data_dict['z'] = data[:,1]/1e3 # in mega meters
    data_dict['T'] = data[:,3]
    data_dict['V'] = data[:,4] # km/s    
    data_dict['P_gas'] = data[:,5]  
    data_dict['P'] = data[:,6] # P_tot = P_gas + 1/2*rho*V^2
    data_dict['n_h'] = data[:,7]
    data_dict['n_hI'] = data[:,8] 
    data_dict['n_e'] = data[:,9] 
    data_dict['n_tot'] = data[:,7] + data[:,9] # n_H + n_e

    return data_dict

def plot_atm(data):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    kb = 1.3806488e-16 # ergs/K

    ax2 = ax.twinx()
    ax.plot(data['z'], data['T'], label='T', color='black', linewidth=2, marker='o')
    ax.set_ylabel('T')
    ax.set_yscale('log')
    
    ax2.plot(data['z'], data['P'], label=r'P$_{\mathrm{tot}}$', linewidth=2, marker='o')
    ax2.plot(data['z'], kb*data['n_tot']*data['T'], label='derived P', linewidth=2, linestyle='dashed')
    ax2.plot(data['z'], data['P_gas'], label='P_gas', linewidth=2, linestyle='dotted')

    ax2.set_yscale('log')
    
    ax2.legend()
    ax.set_xlabel('z [Mm]')
    ax2.set_ylabel('P')
    
    fig.savefig('full_atm.png', dpi=600)
    
    ax.set_xlim(min(data['z']), 2.5)
    fig.savefig('low_atm.png', dpi=600)
    return fig, ax, ax2

def interp(z_in, f_in):
    sorted_i = np.argsort(z_in)
    z = z_in[sorted_i]
    f = f_in[sorted_i]
    print("z {}--{}".format(z[0],z[-1]))
    import scipy.interpolate as scpint
    #int_f = scpint.interp1d(z, f, kind='cubic')
    int_f = scpint.InterpolatedUnivariateSpline(z, f, k=4)
    return int_f

data = read_atm()

for key in data:
    data[key] = data[key][11:129]
    
#print("indices and heights")
#for i, z in enumerate(data['z']):
#    print(i, z)


    
atm_lim = 3.0
fig, ax, axP = plot_atm(data)
z_top = data['z'][0]
z_bot = data['z'][-1]
z_top = atm_lim
nz = 2048
z_basis = de.Chebyshev('z', nz, interval=[z_bot,z_top], dealias=3/2)
domain = de.Domain([z_basis], grid_dtype=np.float64)
P = domain.new_field()
T = domain.new_field()
rho = domain.new_field()
rho2 = domain.new_field()

P_z = domain.new_field()
rho_z = domain.new_field()
del_ln_P = domain.new_field()
del_ln_rho = domain.new_field()
g = domain.new_field()

z = domain.grid(axis=0)
z_dealias = domain.grid(axis=0, scale=domain.dealias)

P['g'] = interp(data['z'],data['P'])(z)
T['g'] = interp(data['z'],data['T'])(z)
mp = 1.6726219e-24
rho['g'] = mp*interp(data['z'],data['n_h'])(z)

kb = 1.3806488e-16 # ergs/K
Na = 6.022140857e23

R = kb*Na # assuming pure hydrogen atmosphere, with molar weight of 1 g/mol
print("gas constant R = {}".format(R))

rho2['g'] = P['g']/(R*T['g'])
P.differentiate('z', out=P_z)
rho.differentiate('z', out=rho_z)
P_z['g']/=1e8 # Mm -> cm
rho_z['g']/=1e8 # Mm -> cm

del_ln_rho.set_scales(domain.dealias, keep_data=True)
del_ln_P.set_scales(domain.dealias, keep_data=True)
del_ln_P['g'] = P_z['g']/P['g']
del_ln_rho['g'] = rho_z['g']/rho['g']
gamma = 5/3 # fixed assumption

def fit_linear(x, b):
    #A = np.array([np.ones(x.shape), x]).T
    A = np.array([x]).T    

    print(A.shape, b.shape)
    Q, R = np.linalg.qr(A)
    solution = np.linalg.solve(R, np.dot(Q.T, b))
    return solution
    
#g.set_scales(domain.dealias, keep_data=False)
#g['g'] = -P_z['g']/rho['g']
g['g'] = 27542.29*(700/(700+z))**(2) # cgs units at solar surface
fit_to_HS = fit_linear(rho['g'], -P_z['g']) # least squares fit; assuming constant g.  cgs units.

print("fit to HS eq for g: {}".format(fit_to_HS))

g.set_scales(1, keep_data=True)
rho.set_scales(1, keep_data=True)
rho2.set_scales(1, keep_data=True)
T.set_scales(1, keep_data=True)
P.set_scales(1, keep_data=True)
del_ln_rho.set_scales(1, keep_data=True)
del_ln_P.set_scales(1, keep_data=True)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(z, P['g'], label='P')
ax.plot(z, T['g'], label='T')
ax.plot(z, g['g'], label='g')
ax.plot(z, rho['g'], label='rho')
#ax.plot(z, rho['g'], label='rho2', linestyle='dashed')

ax.set_yscale('log')
ax.set_xlim(min(z), atm_lim)
ax.legend()

fig.savefig('dedalus_atm.png', dpi=600)

P_z.set_scales(1,keep_data=True)
fig_P = plt.figure()
axP = fig_P.add_subplot(1,1,1)
axP.plot(rho['g'], P_z['g'], label=r'$\nabla P$')
axP.plot(rho['g'], -fit_to_HS*rho['g'], linestyle='dashed', linewidth=2, color='black', label='HS fit, g={:8.3g}'.format(fit_to_HS[0]))
axP.legend(loc='upper right')
axP.set_xlabel(r'$\rho$')
axP.set_ylabel(r'$\nabla P$')
fig_P.savefig("HS_balance.png",dpi=600)

#N2 = g['g']*(1/gamma*del_ln_P['g'] - del_ln_rho['g'])
N2 = fit_to_HS*(1/gamma*del_ln_P['g'] - del_ln_rho['g'])

N = np.sqrt(np.abs(N2))
fig_N2 = plt.figure()
ax = fig_N2.add_subplot(1,1,1)

N_sign = N*np.sign(N2)
plot_abs(z, N_sign*1e3, ax, label=r'$|N| (mhz)$', linewidth=1)
ax.axvline(x=0, linestyle='dashed', color='black', linewidth=2)

ax.legend()
ax2 = ax.twinx()
ax2.plot(z, T['g'], color='black', linewidth=2)
ax2.set_ylabel('T')
ax.set_ylabel('|N| (mhz)') #'N2')
#ax2.set_yscale('log')
ax.set_xlabel('z [Mm]')
ax.set_ylim(0,100)
fig_N2.savefig('brunt.png', dpi=600)
