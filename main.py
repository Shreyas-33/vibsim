# Importing the required packages #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import streamlit as st
np.random.seed(42)
from scipy.optimize import curve_fit
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go


# Input Parameters #
folder_path = "./AI_15676-15677" # Folder path for the chirp up and/or chirp down data
T = 10 # ms - T used for AI
subdirs = os.listdir(folder_path)
basename = os.path.basename(folder_path)
# Checking for subdirectories and reading the fringes_data.csv as pandas dataframe #

if set(['chirp_up','chirp_down']).issubset(subdirs):
    tmpdf_cu = pd.read_csv(folder_path+"/chirp_up/fringes_data.csv")
    tmpdf_cu = tmpdf_cu.sort_values(by="var1")
    # print(tmpdf_cu)
    tmpdf_cd = pd.read_csv(folder_path+"/chirp_down/fringes_data.csv")
    tmpdf_cd = tmpdf_cd.sort_values(by="var1")
    # print(tmpdf_cd)
    print("Both chirp fringes_data.csv file read and stored as dataframes!")
elif set(['chirp_up']).issubset(subdirs):
    tmpdf_cu = pd.read_csv(folder_path+"/chirp_up/fringes_data.csv")
    tmpdf_cu = tmpdf_cu.sort_values(by="var1")
    # print(tmpdf_cu)
    print("Chirp up fringes_data.csv file read and stored as dataframe!")
    tmpdf_cd = None
elif set(['chirp_down']).issubset(subdirs):
    tmpdf_cu = None
    tmpdf_cd = pd.read_csv(folder_path+"/chirp_down/fringes_data.csv")
    tmpdf_cd = tmpdf_cd.sort_values(by="var1")
    # print(tmpdf_cd)
    print("Chirp down fringes_data.csv file read and stored as dataframe!")


st.title("AI Analysis")
st.write(f"Basename: {basename}")

# Defining the fit function for the AI #
def fringe_fit(a,C,g,ct, T=10e-3): # a is chirp rate, C is contrast, g is gravitational acceleration
        c = 299792458
        f1 = 384.2304844685e12-72.9113e6-2.56300597908911e9+61.625e6
        f2 = f1 +6.725e9+112.06936e6
        k1 = 2*np.pi/(c/f1)
        k2 = 2*np.pi/(c/f2)
        keff = k1+k2
        # T = 10e-3
        return (C*np.cos((keff*g-2*np.pi*a)*T**2)+ct)

# Converting required arrays to process the data and fit it#
y_cu = tmpdf_cu["fraction"].to_numpy()
x_cu = tmpdf_cu["var1"].to_numpy() * 1e6  # Convert to MHz/s

# Guess parameters for the fitting #
ymin_idx_cu = tmpdf_cu["fraction"].idxmin()
x_ymin_idx_cu = tmpdf_cu["var1"][ymin_idx_cu]
c = 299792458
f1 = 384.2304844685e12-72.9113e6-2.56300597908911e9+61.625e6
f2 = f1 +6.725e9+112.06936e6
k1 = 2 * np.pi / (c / f1)
k2 = 2 * np.pi / (c / f2)
keff_max = k1 + k2
g_guess = x_ymin_idx_cu * 1e6 * (2 * np.pi / keff_max)  # Convert to MHz/s

# Perform the curve fitting
params_cu, covariance_cu = curve_fit(fringe_fit, x_cu, y_cu, p0=[0.01, g_guess, np.mean(y_cu)])

# Extract fitted parameters
C_fit, g_fit, ct_fit = params_cu

# Calculate uncertainties
errors = np.sqrt(np.diag(covariance_cu))

# Calculate g-value in mGal
g_value_plot = '{:.3f}'.format(g_fit * 1e5)
g_res_plot = '{:.3f}'.format(2 * errors[1] * 1e5)  # g-resolution in mGal
redchisq_plot = '{:.3e}'.format(np.sum((fringe_fit(x_cu, *params_cu) - y_cu)**2 / (y_cu)))

fig = go.Figure()


fig.add_trace(go.Scatter(x=x_cu*1e-6, y=y_cu, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=x_cu*1e-6, y=fringe_fit(x_cu, *params_cu), mode='lines', name='Fit'))


st.plotly_chart(fig, use_container_width=True)
st.write(f"Gravity = {g_value_plot} $\pm$ {g_res_plot} mGal")
st.write(f"Contrast = {abs(2*C_fit)}")

## draw horizontal line in streamlit
st.markdown("""---""")

st.title("Vibration simulation")

T_init = 10e-3
T = float(st.text_input("Enter custom T (ms):", value=f"{T_init*1e3}"))*1e-3

# leftshift = 40          # For 10 ms
# expand = 20             # For 10 ms
leftshift = 0             # For 20 ms
expand = -0                # For 20 ms
start_init = np.min(x_cu- leftshift*25.06)*1e-6 - (leftshift)*25.06*1e-6
end_init = np.max(x_cu)*1e-6 - (leftshift-expand)*25.06*1e-6

# start = float(st.text_input("Enter custom minimum for x-axis (MHz/s):", value=f"{start_init}"))
# end = float(st.text_input("Enter custom maximum for x-axis (MHz/s):", value=f"{end_init}"))

vibration_init = 5
vibration = float(st.text_input("Enter custom vibration noise (mgal):", value=f"{vibration_init}"))
plotlist = []
ideal = []
# contrast_adjust = 0.4
# delta = 0.00125
delta = (vibration*1e-5)/((2*np.pi/keff_max)*1e6)
dof = 3
n_points_init = 200
stdev_init = 0.0022
stdev = float(st.text_input("Enter custom standard deviation:", value=f"{stdev_init}"))

n_points = int(st.text_input("Enter custom number of points:", value=f"{n_points_init}"))
contrast_set_init = -0.15
contrast_set = float(st.text_input("Enter custom contrast:", value=f"{-2*contrast_set_init}"))*-0.5

upper_bound = abs(contrast_set)+ct_fit
lower_bound = ct_fit-abs(contrast_set)

min_value = 24.95
max_value = 25.2
value = [25.05, 25.08] # this is a list of two values
step = 0.0001
slider = st.slider('Select a range for x axis study', min_value, max_value, value, step)
start = slider[0]
end = slider[1]

ideal = fringe_fit(np.linspace(start,end, n_points)*1e6, contrast_set, g_fit, ct_fit, T=T)

def check_bounds(array):
    upper_indices = []
    lower_indices = []
    for i, a in enumerate(array):
        if a > upper_bound:
            upper_indices.append(i)
        elif a < lower_bound:
            lower_indices.append(i)
    
    return upper_indices, lower_indices, len(upper_indices)!=len(lower_indices)

def try_exchange(noise, upper_indices, lower_indices):
    for i in range(min(len(upper_indices), len(lower_indices))):
        up = upper_indices.pop(0)
        low = lower_indices.pop(0)
        noise[up], noise[low] = noise[low], noise[up]

## VIBRATION LOOP ##
if vibration>0:
    for i, a in enumerate(np.linspace(start,end, n_points)):
        current = fringe_fit(a*1e6, contrast_set, g_fit, ct_fit, T=T)
        bounded_points = []
        num_bounded_points = max(200, int((10*delta/((end-start)/25.06))*n_points))
        for num in np.linspace(-delta, delta, num_bounded_points):
            bounded_points.append(fringe_fit((a+num)*1e6, contrast_set, g_fit, ct_fit, T=T))
        
        plotlist.append(random.choice(bounded_points))
        # ideal.append(fringe_fit(a*1e6, contrast_set, g_fit, ct_fit, T=T))

## TYPE 1 NOISE LOOP ##
if stdev!=0:
    while True:
        plotlist_temp = plotlist.copy()
        noise = np.random.normal(0, stdev, n_points)
        plotlist_temp = plotlist_temp+noise

        if check_bounds(plotlist_temp)[2]:
            continue

        if len(check_bounds(plotlist_temp)[0]) > 0:
            try_exchange(noise, check_bounds(plotlist_temp)[0], check_bounds(plotlist_temp)[1])
        break

    plotlist = plotlist_temp

params, cov = curve_fit(lambda x, contrast, g, ct: fringe_fit(x*1e6, contrast, g, ct, T=T), np.linspace(start,end, n_points), plotlist, p0=[contrast_set, g_fit, ct_fit])

errors = np.sqrt(np.diag(cov))
g_value_plot = '{:.3f}'.format(params[1] * 1e5)
g_res_plot = '{:.3f}'.format(2 * errors[1] * 1e5)  # g-resolution in mGal
redchisq_plot = '{:.3e}'.format(np.sum((fringe_fit(np.linspace(start, end, n_points)*1e6, *params, T=T) - np.array(plotlist))**2 / (np.array(plotlist))))


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()


ax1.plot(np.linspace(start,end, n_points), ideal, label="Ideal before adding noise", color="red")
ax1.plot(np.linspace(start,end, n_points), fringe_fit(np.linspace(start,end, n_points)*1e6, *params, T=T), label="Noise fit", color="orange")
ax2.set_xlim(np.array(ax1.get_xlim())*(2*np.pi/keff_max)*1e5*1e6)
ax2.set_xlabel("Gravitational Acceleration (mGal)")
ax1.set_xlabel("Chirp Rate (MHz/s)")
ax2.axvline(x=params[1]*1e5, color="green", linestyle="--", label="g-value")
ax1.scatter(np.linspace(start,end, n_points), plotlist, alpha=0.5, label=f"{n_points} Noise added data points")
plt.figtext(0.05, -0.14, "$\sigma_{z}$ = "+str(delta*(2*np.pi/keff_max)*1e5*1e6)+" mgal"+"\nNoise added to the data with stdev: "+str(stdev)+"\nGravity ideal = "+str(g_fit*1e5)+" mGal"+"\nGravity noise = "+str(g_value_plot)+" $\pm$ "+str(g_res_plot)+" mGal\n$\chi^2_{red}$ = "+str(redchisq_plot)+"\nContrast of ideal line = "+str(abs(2*contrast_set))+"\nContrast of noise line = "+'{:.3f}'.format(2*params[0]));
ax1.legend()
plt.title("Vibration Noise, T = "+ str(T*1e3)+ " ms");

st.pyplot(fig)

# st.markdown("""---""")

st.title("Multiple runs errorchart [$\pm$ 2$\sigma$]]")

gvals = []
gerrors = []
n_runs_init = 100
n_runs = int(st.text_input("Enter custom number of runs:", value=f"{n_runs_init}"))

for i in range(n_runs):
    plotlist = []
    ideal = []

    for i, a in enumerate(np.linspace(start,end, n_points)):
        current = fringe_fit(a*1e6, contrast_set, g_fit, ct_fit, T=T)
        bounded_points = []
        for num in np.linspace(-delta, delta, int((10*delta/0.012)*n_points)):
            bounded_points.append(fringe_fit((a+num)*1e6, contrast_set, g_fit, ct_fit, T=T))
        
        plotlist.append(random.choice(bounded_points))
        ideal.append(fringe_fit(a*1e6, contrast_set, g_fit, ct_fit, T=T))

    params, cov = curve_fit(lambda x, contrast, g, ct: fringe_fit(x*1e6, contrast, g, ct, T=T), np.linspace(start,end, n_points), plotlist, p0=[contrast_set, g_fit, ct_fit])

    errors = np.sqrt(np.diag(cov))
    g_value_plot = '{:.3f}'.format(params[1] * 1e5)
    g_res_plot = '{:.3f}'.format(2 * errors[1] * 1e5)  # g-resolution in mGal
    redchisq_plot = '{:.3e}'.format(np.sum((fringe_fit(np.linspace(start, end, n_points)*1e6, *params, T=T) - np.array(plotlist))**2 / (np.array(plotlist))))

    gvals.append(float(g_value_plot))
    gerrors.append(float(g_res_plot))

fig = plt.figure(figsize=(12,8))
plt.errorbar(range(n_runs), gvals, yerr=gerrors, fmt='o')
plt.hlines(g_fit*1e5, 0, n_runs, color="red", label="Ideal g-value")
plt.ylim(0.99999*g_fit*1e5, 1.00001*g_fit*1e5)
plt.xlabel("Number of iterations")
plt.ylabel("Gravitational Acceleration (mGal)")
plt.title(f"Vibration Noise over {n_runs} iterations, No of points in fringe = {n_points}, T={T*1e3} ms, Contrast = "+str(abs(2*contrast_set))+ ", $\sigma_{z}$ = "+str(delta*(2*np.pi/keff_max)*1e5*1e6)+" mgal")

st.pyplot(fig)