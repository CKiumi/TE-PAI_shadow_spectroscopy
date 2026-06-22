import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# Global variables for font sizes
Title_fontsize = 18
Label_fontsize = 14
Legend_fontsize = 12
tick_labelsize = 12
colors=['red', 'blue', 'green', 'purple', 'orange']

def save_fig(file_name: str, folder: str) -> None:
    file_name = r'{0}{1}'.format(
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_"), file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, file_name) + ".png")
    print(f"Figure saved in {folder}/{file_name}.png")


def plot_double_histo(x1: np.ndarray, x2: np.ndarray, bins1=50, bins2=50, save_as=None, Folder="Data") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(x1, bins=bins1, color='blue', edgecolor='black')
    ax1.set_title("Histogram of real part of o", fontsize=Title_fontsize)
    ax1.set_xlabel("o_real", fontsize=Label_fontsize)
    ax1.set_ylabel("count", fontsize=Label_fontsize)
    ax2.hist(x2, bins=bins2, color='red', edgecolor='black')
    ax2.set_title("Histogram of imaginary part of o", fontsize=Title_fontsize)
    ax2.set_xlabel("o_im", fontsize=Label_fontsize)
    ax2.set_ylabel("count", fontsize=Label_fontsize)
    plt.tight_layout()
    if isinstance(save_as, str):
        save_fig(save_as, Folder)


def plot_histo(val: np.ndarray, bins=30, title='Histogram of values', x_label='values', line=None, save_as=None, Folder="Data") -> None:
    fig, ax = plt.subplots()
    hist, bins = np.histogram(np.real(val), bins=bins)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, width=width)
    val_mean = np.mean(val)
    val_std = np.std(val)
    ax.axvline(val_mean, color='r', linestyle='--', linewidth=1,
               label=f'Expectation: {val_mean:.4f}\nStd: {val_std:.4f}')
    if line is not None:
        ax.axvline(line, color='b', linestyle='--',
                   linewidth=1, label=f'Ideal: {line:.4f}')
    ax.set_title(title, fontsize=Title_fontsize)
    ax.set_xlabel(x_label, fontsize=Label_fontsize)
    ax.set_ylabel("count", fontsize=Label_fontsize)
    ax.legend(fontsize=Legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    if isinstance(save_as, str):
        save_fig(save_as, Folder)


def plot_matrix(matrix: np.ndarray, title: str = "Matrix", x_label: str = "", y_label: str = "", save_as=None, Folder="Data") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=Title_fontsize)
    ax.set_xlabel(x_label, fontsize=Label_fontsize)
    ax.set_ylabel(y_label, fontsize=Label_fontsize)
    if isinstance(save_as, str):
        save_fig(save_as, Folder)


def plot_data(x:np.ndarray,y:np.ndarray,
              mean=True,max=True, min=True,connect=False, 
              title="",x_label="",y_label="",save_as=None, Folder="Data"):
    
    """Plot the data in a scatter plot

    Args:
        x (np.ndarray): x array for the plot
        y (np.ndarray): y array for the plot
        mean (bool, optional): If True add an horizontal line at the y mean. Defaults to True.
        max (bool, optional): If True add an horizontal line at the y max. Defaults to True.
        min (bool, optional): If True add an horizontal line at the y min. Defaults to True.
        connect (bool, optional): If True, connect all the points from the scatter plot. Defaults to False.
        title (str, optional): Title_. Defaults to "".
        x_label (str, optional): x_label. Defaults to "".
        y_label (str, optional): y_label. Defaults to "".
        save_as (str, optional):  if not empty save file with name = input str. Defaults to "".
        Folder (str, optional): folder to save the figure. Defaults to "Data".
        Returns:
        _type_: None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x,y, color='black')
    if connect:
        plt.plot(x,y, color='blue')
    if mean:
        ymean=np.mean(y)
        plt.axhline(y=ymean, color='green', linestyle='--', label=f"Average = {ymean:.2f}")
        plt.text(len(y)-1, mean, f"{mean:.2f}", color='green', fontsize=10, va='center', ha='left')
    if max:
        ymax=np.amax(y)
        plt.axhline(y=ymax, color='red', linestyle='--', label=f"Max = {ymax:.2f}")
        plt.text(len(y)-1, ymax, f"{ymax:.2f}", color='red', fontsize=10, va='center', ha='left')
    if min:
        ymin=np.amin(y)
        plt.axhline(y=ymin, color='orange', linestyle='--', label=f"Min = {ymin:.2f}")
        plt.text(len(y)-1, ymin, f"{ymin:.2f}", color='orange', fontsize=10, va='center', ha='left')

    # Ajouter des labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # Save the plot if save_as is provided
    if isinstance(save_as, str):
         save_fig(save_as,Folder)


def plot_spectre(frequencies: np.ndarray, values: np.ndarray, Energy_gap: list[float] = None,
                 title: str = "Spectral Cross-Correlation", label: str = "Shadow spectroscopy",
                 save_as: str = None, Folder: str = "Data"):
    pos = np.argsort(np.abs(values))[::-1]
   
    
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, values, color='blue', label=label)
    # plt.axvline(x=frequencies[pos[3]], color='black', linestyle='--',
    #                 linewidth=1, label=f"Real Energy Gap: {frequencies[pos[2]]:.2f} rad/s")
    # plt.axvline(x=frequencies[pos[4]], color='black', linestyle='--',
    #                 linewidth=1, label=f"Real Energy Gap: {frequencies[pos[3]]:.2f} rad/s")
    if isinstance(Energy_gap, list):
        for i, energy_gap in enumerate(Energy_gap):
            plt.axvline(x=energy_gap, color='black', linestyle='--',
                        linewidth=1, label=f"Real Energy Gap: {energy_gap:.2f} rad/s")
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=Title_fontsize)
    plt.xlabel("Frequency (rad/s)", fontsize=Label_fontsize)
    plt.ylabel("Amplitude,\n Arbitrary units", fontsize=Label_fontsize)
    plt.legend(fontsize=Legend_fontsize, loc="best")
    if isinstance(save_as, str):
        save_fig(save_as, Folder)


def plot_multiple_data(X_list: list[list], Y_list: list[list],
                       labels: list[str], x_label: str = "Frequency",
                       y_label: str = "Amplitude,\n Arbitrary units", title: str = "Spectral cross correlation",
                       color: list =["red","green","blue"],
                       linestyle: list =  ['-','-', '-'],
                       axvline: list = [],
                       save_as: str = None, Folder: str = "Data"):

    freq_max = []
    plt.figure(figsize=(10, 6))
    for i in range(len(X_list)):
        plt.plot(X_list[i], Y_list[i], label=labels[i],  color=color[i%len(color)], linestyle=linestyle[i%len(linestyle)])
        pos_max = np.argsort(np.abs(Y_list[i]))[::-1]
        freq_max.append(X_list[i][pos_max[:len(axvline)]])
    for i in range(len(axvline)):
        plt.axvline(x=axvline[i], color='gray', linestyle='--', linewidth=1,
                    label=f"Theoretical Energy Gap : {axvline[i]:.3f}")
        plt.text(axvline[i] * 0.41, np.max(X_list) * (1 + 0.05), f"{axvline[i]:.3f}",
                 color='black', ha='left', va='center', fontsize=9)
        

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    # 'both' applies to x and y axis
    plt.tick_params(axis='both', labelsize=tick_labelsize)
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=Title_fontsize, fontweight='bold')
    plt.xlabel(x_label, fontsize=Label_fontsize)
    plt.ylabel(y_label, fontsize=Label_fontsize)
    plt.legend(fontsize=10, loc="best")

    # Save the plot if save_as is provided
    if isinstance(save_as, str):
        save_fig(save_as, Folder)


def plot_semi_log_data(x: np.ndarray, y: np.ndarray, linthresh : float=1e-3,
            title: str = "Semi-log plot", x_label: str = "x", y_label: str = "y",scatter: bool = False,
            save_as: str = None, Folder: str = "Data") -> None:
    
    plt.figure(figsize=(8, 5))
    if scatter:
        plt.scatter(x, y, color='blue')
    else:
        plt.plot(x, y, color='blue')
        
    plt.yscale('symlog', linthresh=1e-3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    # 'both' applies to x and y axis
    plt.tick_params(axis='both', labelsize=tick_labelsize)
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.title(title, fontsize=Title_fontsize, fontweight='bold')
    plt.xlabel(x_label, fontsize=Label_fontsize)
    plt.ylabel(y_label, fontsize=Label_fontsize)
    plt.legend(fontsize=10, loc="best")

    # Save the plot if save_as is provided
    if isinstance(save_as, str):
        save_fig(save_as, Folder)

