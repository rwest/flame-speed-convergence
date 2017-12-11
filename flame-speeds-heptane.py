
# coding: utf-8

# # Laminar flame speed calculations (n-heptane)

# In[1]:

import os
import cantera as ct
import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import re
import scipy
import scipy.optimize


# In[2]:

model_name = 'heptane'
fuel = 'NC7H16'


# In[3]:

cantera_file_path = model_name+'.cti'
print(cantera_file_path)
assert os.path.exists(cantera_file_path)
gas = ct.Solution(cantera_file_path)


# In[4]:

phi = 1.0
temperature = 298 # K
pressure = 1 * ct.one_atm
print("T = {} K".format(temperature))
print("P = {} Pa".format(pressure))
gas.set_equivalence_ratio(phi, fuel, 'O2:1.0, N2:3.76')
gas.mole_fraction_dict()
gas.TP = (temperature,pressure)
gas()


# In[5]:

def extrapolate_uncertainty(grids, speeds):
    """
    Given a list of grid sizes and a corresponding list of flame speeds,
    extrapolate and estimate the uncertainty in the final flame speed.
    Also makes a plot.
    """
    grids = list(grids)
    speeds = list(speeds)
    def speed_from_grid_size(grid_size, true_speed, error):
        """
        Given a grid size (or an array or list of grid sizes)
        return a prediction (or array of predictions)
        of the computed flame speed, based on 
        the parameters `true_speed` and `error`
        """
        return true_speed +  error * np.array(grid_size)**-1.

    popt, pcov = scipy.optimize.curve_fit(speed_from_grid_size, grids[-4:], speeds[-4:])

    perr = np.sqrt(np.diag(pcov))
    true_speed  = popt[0]
    percent_error_in_true_speed = 100.*perr[0] / popt[0]
    print("Fitted true_speed is {:.4f} ± {:.4f} cm/s ({:.1f}%)".format(
        popt[0]*100,
        perr[0]*100,
        percent_error_in_true_speed
        ))
    #print "convergerce rate wrt grid size is {:.1f} ± {:.1f}".format(popt[2], perr[2])
    estimated_percent_error = 100. * (speed_from_grid_size(grids[-1], *popt) - true_speed) / true_speed
    print("Estimated error in final calculation {:.1f}%".format(estimated_percent_error))

    total_error_estimate = abs(percent_error_in_true_speed) + abs(estimated_percent_error)
    print("Estimated total error {:.1f}%".format(total_error_estimate))

    plt.semilogx(grids,speeds,'o-')
    plt.ylim(min(speeds[-5:]+[true_speed-perr[0]])*.95, max(speeds[-5:]+[true_speed+perr[0]])*1.05)
    plt.plot(grids[-4:], speeds[-4:], 'or')
    extrapolated_grids = grids + [grids[-1] * i for i in range(2,8)]
    plt.plot(extrapolated_grids,speed_from_grid_size(extrapolated_grids,*popt),':r')
    plt.xlim(*plt.xlim())
    plt.hlines(true_speed, *plt.xlim(), colors=u'r', linestyles=u'dashed')

    plt.hlines(true_speed+perr[0], *plt.xlim(), colors=u'r', linestyles=u'dashed', alpha=0.3)
    plt.hlines(true_speed-perr[0], *plt.xlim(), colors=u'r', linestyles=u'dashed', alpha=0.3)
    plt.fill_between(plt.xlim(), true_speed-perr[0],true_speed+perr[0], facecolor='red', alpha=0.1 )

    #plt.text(grids[-1],speeds[-1],"{:.1f}%".format(estimated_percent_error))

    above = popt[1]/abs(popt[1]) # will be +1 if approach from above or -1 if approach from below
    
    plt.annotate("",
                xy=(grids[-1], true_speed),
                 xycoords='data',
                xytext=(grids[-1], speed_from_grid_size(grids[-1], *popt)),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='|-|',
                                connectionstyle='arc3',
                                color='black', shrinkA=0, shrinkB=0),
                )
        
    plt.annotate("{:.1f}%".format(abs(estimated_percent_error)),
                xy=(grids[-1], speed_from_grid_size(grids[-1], *popt)),
                 xycoords='data',
                xytext=(10,20*above),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3')
                )
    
    plt.annotate("",
                xy=(grids[-1]*4, true_speed-(above*perr[0])),
                 xycoords='data',
                xytext=(grids[-1]*4, true_speed),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='|-|',
                                connectionstyle='arc3',
                                color='black', shrinkA=0, shrinkB=0),
                )
    plt.annotate("{:.1f}%".format(abs(percent_error_in_true_speed)),
                xy=(grids[-1]*4, true_speed-(above*perr[0])),
                 xycoords='data',
                xytext=(10,-20*above),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3')
                )

    plt.ylabel("Flame speed (m/s)")
    plt.xlabel("Grid size")
    plt.show()
    
    return true_speed, total_error_estimate


# In[6]:

def make_callback(flame):
    speeds = []
    grids = []

    def callback(_):
        speed = flame.u[0]
        grid = len(flame.grid)
        speeds.append(speed)
        grids.append(grid)
        print("Iteration {}".format(len(grids)))
        print("Current flame speed is is {:.4f} cm/s".format(speed*100.))
        if len(grids) < 5:
            return 1.0 # 
        try:
            extrapolate_uncertainty(grids, speeds)
        except Exception as e:
            print("Couldn't estimate uncertainty", e.message)
            return 1.0 # continue anyway

        return 1.0
    return callback, speeds, grids

# flame.set_steady_callback(make_callback()[0])


# In[7]:

# Domain width in metres
width = 0.015

# Create the flame object
flame = ct.FreeFlame(gas, width=width)

# Define tolerances for the solver
# (these are used throughout the notebook)
#refine_criteria = {'ratio':3, 'slope': 0.1, 'curve': 0.1}
refine_criteria = {'ratio':2, 'slope': 0.05, 'curve': 0.05}
flame.set_refine_criteria(**refine_criteria)
flame.set_max_grid_points(flame.domains[flame.domain_index('flame')], 1e4)

callback, speeds, grids = make_callback(flame)
flame.set_steady_callback(callback)

# Define logging level
loglevel = 1


# In[8]:

flame.solve(loglevel=loglevel, auto=True)


# In[9]:

final_true_speed, percentage_uncertainty = extrapolate_uncertainty(grids, speeds)


# In[10]:

print("Final grid was size {}".format(grids[-1]))
print("Final speed was {:.4f} cm/s".format(100*speeds[-1]))
print("Estimated uncertainty is {:.1f}%".format(percentage_uncertainty))
print("i.e. {:.3f} +/- {:.3f} cm/s".format(100*speeds[-1],
                                           percentage_uncertainty*speeds[-1]))


# In[11]:

for i in range(4,len(grids)):
    print("At step {}".format(i))
    print("Grid was size {}".format(grids[i]))
    print("Speed was {:.4f} cm/s".format(100*speeds[i]))
    true_speed, percentage_uncertainty = extrapolate_uncertainty(grids[:i], speeds[:i])
    print("Estimated uncertainty was {:.1f}%".format(percentage_uncertainty))
    print("i.e. {:.3f} +/- {:.3f} cm/s".format(100*speeds[i],
                                           percentage_uncertainty*speeds[i]))
    print("or  {:.3f} -- {:.3f} cm/s".format((100-percentage_uncertainty)*speeds[i],
                                           (100+percentage_uncertainty)*speeds[i]))
    print("(For reference, the 'final' extrapolated speed was {:.3f} cm/s".format(100*final_true_speed))
    print("="*80)


# And plot some figures just to check things look sensible.

# In[12]:

plt.figure()
plt.plot(flame.grid*100, flame.T, '-o')
plt.xlabel('Distance (cm)')
plt.ylabel('Temperature (K)');


# In[13]:

# look up species indices
i_fuel = gas.species_index(fuel)
i_co2 = gas.species_index('CO2')
i_h2o = gas.species_index('H2O')

# Extract concentration data
X_FUEL = flame.X[i_fuel]
X_CO2 = flame.X[i_co2]
X_H2O = flame.X[i_h2o]

plt.figure()

plt.plot(flame.grid*100, X_FUEL, '-', label=r'n-$C_{7}H_{16}$')
plt.plot(flame.grid*100, X_CO2, '-', label=r'$CO_{2}$')
plt.plot(flame.grid*100, X_H2O, '-', label=r'$H_{2}O$')
plt.legend(loc=2)
plt.xlabel('Distance (cm)')
plt.ylabel('MoleFractions');


# In[ ]:



