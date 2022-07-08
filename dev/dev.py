def plot_landscape(threshold_mode, losses=losses, grad_P_dict=grad_P_dict, grad_bias_dict=grad_bias_dict):
    XX, YY = np.meshgrid(all_P_bise, all_bias_bise)


    fig, axs = plt.subplots(3, 3, subplot_kw={"projection": "3d"}, figsize=(8, 8))
    fig.suptitle(f'threshold: {threshold_mode}')

#     idxs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    keys = list(losses.keys())

    for idx in range(len(keys)):
#         ax_idx = idxs[idx]
        surf = axs[idx, 0].plot_surface(XX, YY, values[threshold_mode, keys[idx]], label=keys[idx])
        axs[idx, 0].set_xlabel('P')
        axs[idx, 0].set_ylabel('bias')
        axs[idx, 0].set_title(f'loss {keys[idx]}')
    
    for idx in range(len(keys)):
#         ax_idx = idxs[idx]
        surf = axs[idx, 1].plot_surface(XX, YY, grad_P_dict[threshold_mode, keys[idx]], label=keys[idx])
        axs[idx, 1].set_xlabel('P')
        axs[idx, 1].set_ylabel('bias')
        axs[idx, 1].set_title(f'grad P {keys[idx]}')

    for idx in range(len(keys)):
#         ax_idx = idxs[idx]
        surf = axs[idx, 2].plot_surface(XX, YY, grad_bias_dict[threshold_mode, keys[idx]], label=keys[idx])
        axs[idx, 2].set_xlabel('P')
        axs[idx, 2].set_ylabel('bias')
        axs[idx, 2].set_title(f'grad bias {keys[idx]}')
        
    plt.show()