import numpy as np
from scipy.optimize import curve_fit


# Define the sigmoid function
def sigmoid(T, T_m, k):
    return 1 / (1 + np.exp(-(T - T_m) / k))

def get_Tm(T, f_T, p0=[500, -10], get_err=False):
    # Fit the data
    popt, pcov = curve_fit(sigmoid, T, f_T, p0=p0)  # Initial guesses for T_m and k
    # Extract fitted parameters
    T_m, k = popt
    if not get_err:
        return T_m, k
    else:
        return (T_m, k), pcov

def b_sigmoid(T, Tm, slope, top, bottom):
    return bottom + (top - bottom)/(1 + np.exp(-(T - Tm)/slope))

def fit_hTm(T, f_T, p0=[500, -10]):
    top = np.max(f_T)
    bottom = np.min(f_T)
    data_sigmoid = lambda T, Tm, slope: b_sigmoid(T, Tm, slope, top=top, bottom=bottom)
    popt, pcov = curve_fit(data_sigmoid, T, f_T, p0=p0)  # Initial guesses for T_m and k
    # Extract fitted parameters
    Tm, slope = popt
    fitted_sigmoid = lambda T: b_sigmoid(T, Tm=Tm, slope=slope, top=top, bottom=bottom)
    return Tm, slope, fitted_sigmoid

def plot_sigmoid(
        ax, T, f_T, color, use_b_fit=True, p0=[500, -10], n_plot_points=250,
        label=None, use_label=True
    ):
    if not use_b_fit:
        Tm, k = get_Tm(T, f_T, p0=p0)
        # f_T = (f_T - f_T.min())/(f_T.max() - f_T.min())
        T_fit = np.linspace(min(T), max(T), n_plot_points)
        f_T_fit = sigmoid(T_fit, Tm, k)
        # f_fit = (f_T.max() - f_T.min())*f_fit + f_T.min()
    else:
        Tm, _, fitted_sigmoid = fit_hTm(T, f_T)
        T_fit = np.linspace(min(T), max(T), n_plot_points)
        f_T_fit = fitted_sigmoid(T_fit)
    if use_label:
        if label is None:
            label = rf"$\hat{{T}}_m$={Tm:.0f} K"
    else:
        label = None
    plot_s = ax.plot(
        T_fit, f_T_fit, color=color, alpha=0.5, ls="--",
        label=label
    )
    return Tm