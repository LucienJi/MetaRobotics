import matplotlib.pyplot as plt 
import seaborn  as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.lines import Line2D
def sliding_window_average_filter(data, window_size):
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        for t in range(data.shape[0]):
            start = max(0, t - window_size // 2)
            end = min(data.shape[0], t + window_size // 2 + 1)
            filtered_data[t, i] = np.mean(data[start:end, i])

    return filtered_data

def plot_bar_chart(data, data_sources=None, title=None):
    n, num_data = data.shape
    x = np.arange(n)

    mean_values = np.mean(data, axis=1)
    std_values = np.std(data, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, mean_values, yerr=std_values, align='center', alpha=0.7, ecolor='black', capsize=5)

    ax.set_xlabel('Data Sources')
    ax.set_ylabel('Mean and Standard Deviation')
    ax.set_title(title or 'Bar Chart')

    if data_sources is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(data_sources, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_vel(steps, lin_vel=None, command_vel=None, ang_vel=None,
             COT=None, stride_variability=None, args=None, macro_steps=None,
             save_path=None, legend=True, title=None, fontsize=12,
             lin_vel_colors=None, command_vel_colors=None, ang_vel_color=None,
             filter_window_size=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    if lin_vel is not None:
        if filter_window_size is not None:
            lin_vel = sliding_window_average_filter(lin_vel, filter_window_size)

        lin_vel_colors = lin_vel_colors or ['tab:blue', 'tab:orange']
        ax.plot(steps, lin_vel[:, 0], label='Vel x', color=lin_vel_colors[0], marker='o')
        ax.plot(steps, lin_vel[:, 1], label='Vel y', color=lin_vel_colors[1], marker='o')

    if command_vel is not None:
        if filter_window_size is not None:
            command_vel = sliding_window_average_filter(command_vel, filter_window_size)

        command_vel_colors = command_vel_colors or ['tab:red', 'tab:green', 'tab:blue']
        ax.plot(steps, command_vel[:, 0], label='Vel x', color=command_vel_colors[0], marker='o')
        ax.plot(steps, command_vel[:, 1], label='Vel y', color=command_vel_colors[1], marker='o')
        ax.plot(steps, command_vel[:, 2], label='Ang Vel', color=command_vel_colors[2], marker='o')

    if ang_vel is not None:
        if filter_window_size is not None:
            ang_vel = sliding_window_average_filter(ang_vel.reshape(-1, 1), filter_window_size)
            ang_vel = ang_vel.flatten()

        ang_vel_color = ang_vel_color or 'tab:purple'
        ax.plot(steps, ang_vel, label='Ang Vel', color=ang_vel_color, marker='o')

    if COT is not None:
        if filter_window_size is not None:
            COT = sliding_window_average_filter(COT.reshape(-1, 1), filter_window_size)
            COT = COT.flatten()

        ax.scatter(macro_steps, COT, label='Cost of Transport (COT)', marker='o', color='tab:brown')

    if stride_variability is not None:
        if filter_window_size is not None:
            stride_variability = sliding_window_average_filter(stride_variability.reshape(-1, 1), filter_window_size)
            stride_variability = stride_variability.flatten()

        ax.scatter(macro_steps, stride_variability, label='Stride Variability', marker='o', color='tab:pink')

    ax.set_xlabel('Step')
    ax.set_ylabel('Values')

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if legend:
        ax.legend()


    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def sliding_window_average_filter(contacts, window_size):
    filtered_contacts = np.zeros_like(contacts, dtype=bool)
    for i in range(contacts.shape[1]):
        for t in range(contacts.shape[0]):
            start = max(0, t - window_size // 2)
            end = min(contacts.shape[0], t + window_size // 2 + 1)
            filtered_contacts[t, i] = np.mean(contacts[start:end, i]) > 0.5

    return filtered_contacts

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_stability(body_height, body_raw, body_pitch, env_names=None, title=None, dt=1.0, 
                   xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=14, save_path=None):
    # Check if input arrays have the same shape
    if body_height.shape != body_raw.shape or body_height.shape != body_pitch.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    n_env, n_step = body_height.shape
    
    # Create a color map for different environments
    colors = plt.cm.viridis(np.linspace(0, 1, n_env))
    
    # Create subplots for each data
    fig, axs = plt.subplots(3, 1, figsize=(12,8), sharex=True)

    # Plot body_height
    for env in range(n_env):
        axs[0].plot(np.arange(n_step) * dt, body_height[env], label=env_names[env] if env_names else f'Env {env}', color=colors[env])
    axs[0].set_ylabel('Vel. Z (m/s)', fontsize=ylabel_fontsize)
    
    # Plot body_raw
    for env in range(n_env):
        axs[1].plot(np.arange(n_step) * dt, body_raw[env], label=env_names[env] if env_names else f'Env {env}', color=colors[env])
    axs[1].set_ylabel('Vel. Raw (rad/s)', fontsize=ylabel_fontsize)
    
    # Plot body_pitch
    for env in range(n_env):
        axs[2].plot(np.arange(n_step) * dt, body_pitch[env], label=env_names[env] if env_names else f'Env {env}', color=colors[env])
    axs[2].set_xlabel('Time', fontsize=xlabel_fontsize)
    axs[2].set_ylabel('Vel. Pitch (rad/s)', fontsize=ylabel_fontsize)
    
    # Share a single legend for all subplots at the right
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set a custom title if provided
    if title:
        plt.suptitle(title, fontsize=title_fontsize,x = 0.45)

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        # Display the plot if no save path is provided
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

def generate_gradient_cmap(start_color, end_color):
    colors = [start_color, end_color]
    n_bins = 100
    cmap_name = f'custom_{start_color}_{end_color}'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm

def visualize_legged_robot_contacts(contacts, args=['LF', 'RF', 'LR', 'RR'],
                                    xlim=None, title=None, title_fontsize=12, dt = 0.02,
                                    xlabel_fontsize=10, save_path=None, filter_window_size=None):
    # Apply filtering if filter_window_size is provided
    if filter_window_size is not None:
        contacts = sliding_window_average_filter(contacts, filter_window_size)

    # Define gradient colors for each leg
    gradient_colors = {
        'LF': generate_gradient_cmap('white', 'r'),
        'RF': generate_gradient_cmap('white', 'b'),
        'LR': generate_gradient_cmap('white', 'green'),
        'RR': generate_gradient_cmap('white', 'grey')
    }

    # Create a new figure with subplots
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 6))

    for leg in range(contacts.shape[1]):
        leg_contact = contacts[:, leg]  # Extract contact values for the leg

        leg_name = args[leg] if leg < len(args) else f"Leg {leg+1}"
        leg_cmap = gradient_colors.get(leg_name, plt.cm.gray)  # Use gray colormap if leg name not found

        # Plot contact values using imshow and custom colormap
        im = axes[leg].imshow([leg_contact], cmap=leg_cmap, aspect='auto', extent=[0, contacts.shape[0]* dt, 0, 1])

        # Set subplot title and y-axis labels
        # axes[leg].set_title(f"{leg_name} Contact", fontsize=title_fontsize)
        axes[leg].set_ylabel(f"{leg_name} Contact", fontsize=title_fontsize)
        axes[leg].set_yticks([])
        # axes[leg].set_yticklabels(['False', 'True'], fontsize=xlabel_fontsize)

    # Set x-axis label and limits
    axes[-1].set_xlabel("Time", fontsize=xlabel_fontsize)
    if xlim is not None:
        axes[-1].set_xlim(xlim)

    # Create a colorbar for the colormap
    # cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.05)
    # cbar.set_ticks(np.linspace(0, 1, len(args)))
    # cbar.set_ticklabels(args)

    # Adjust subplot layout
    plt.tight_layout()

    # Set overall figure title
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize, y=0.95)

    # Show or save the figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def return_map(states, names, label, title=None, save_path=None, fontsize=12, x_unit='', y_unit=''):
    n_steps, n_env = states.shape

    # Calculate the number of rows and columns for subplot arrangement
    n_cols = int(np.ceil(np.sqrt(n_env)))
    n_rows = int(np.ceil(n_env / n_cols))
    
    # Create a figure with subplots for each environment
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust spacing between subplots
    for env in range(n_env):
        ax = axes[env // n_cols, env % n_cols]  # Get the current subplot axis
        x = states[:-1, env]  # State at time t
        y = states[1:, env]   # State at time t+1
        clf = LocalOutlierFactor(n_neighbors=2) 
        state = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
        y_pred = clf.fit_predict(state)
        x = x[y_pred==1]
        y = y[y_pred==1]

        ax.scatter(x, y, label=names[env] if names else f'Env {env + 1}')
        ax.set_xlabel(f'{label}'+'$_t$' + f'({x_unit})', fontsize=fontsize)
        ax.set_ylabel(f'{label}'+ '$_{t+1}$'+ f'({y_unit})', fontsize=fontsize)
        ax.set_title(names[env] if names else f'Env {env + 1}', fontsize=fontsize)

        # Set the same limits for both axes
        x_mean = x.mean()
        y_mean = y.mean()
        ax.set_xlim(x.min() - 0.2 * x_mean, x.max()+ 0.2 * x_mean)
        ax.set_ylim(y.min() - 0.2 * y_mean, y.max()+ 0.2 * y_mean)

    # Set a common title with LaTeX formatting if provided
    if title:
        plt.suptitle(rf'{title}', fontsize=fontsize + 2, y = 0.95)

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        # Display the return map plot
        plt.show()

def recurrence_plot(body_states, title, env_names, title_fontsize=14, label_fontsize=12, save_path=None, dt = 0.02):
    n_step, n_env, n_dims = body_states.shape

    # Normalize the states for each environment
    normalized_states = np.zeros_like(body_states)
    for env in range(n_env):
        for dim in range(n_dims):
            max_val = np.max(body_states[:, env, dim])
            min_val = np.min(body_states[:, env, dim])
            mean_val =( max_val + min_val)/ 2.0 
            r_val = max_val - min_val
            normalized_states[:, env, dim] = (body_states[:, env, dim] - mean_val) /r_val

    # Calculate the pairwise Euclidean distances between states for each environment
    distances = np.ones((n_env, n_step, n_step))
    for env in range(n_env):
        for t1 in range(n_step):
            for t2 in range(n_step):
                distances[env, t1, t2] = np.sum(np.abs(normalized_states[t1, env] - normalized_states[t2, env]))

    # Calculate the optimal subplot arrangement
    n_cols = int(np.ceil(np.sqrt(n_env)))
    n_rows = int(np.ceil(n_env / n_cols))

    # Create subplots for each environment
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8), sharex='col', sharey='row')
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

    for env in range(n_env):
        ax = axes[env // n_cols, env % n_cols]

        # Create the recurrence plot for the current environment
        
        rotated_distace = np.rot90(distances[env],axes = (0,1))
        im = ax.imshow(rotated_distace, cmap='viridis', extent=[0, n_step * dt, 0, n_step * dt ], aspect='auto')
        ax.set_title(env_names[env], fontsize=label_fontsize)
        if env // n_cols == n_rows - 1:
            ax.set_xlabel('Time(s)', fontsize=label_fontsize)
        if env % n_cols == 0:
            ax.set_ylabel('Time(s)', fontsize=label_fontsize)

    # Add colorbar at the bottom of the figure
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.01])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('$|x_i - x_j|$', fontsize=label_fontsize)

    # Adjust subplot layout
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_3d_points(points, other_points=None, x_label='x', y_label='y', z_label='z',
                   title='3D Scatter Plot', marker='o', marker_size=50,
                   marker_color='blue', other_marker='o', other_marker_size=50,
                   other_marker_color='red', xlim=None, ylim=None, zlim=None,
                   viewpoint=(30,15), save_path=None, title_fontsize=18, label_fontsize=16):
    sns.set(style='whitegrid')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    ax.scatter(x, y, z, marker=marker, s=marker_size, c=marker_color)

    if other_points is not None:
        other_x = [point[0] for point in other_points]
        other_y = [point[1] for point in other_points]
        other_z = [point[2] for point in other_points]
        ax.scatter(other_x, other_y, other_z, marker=other_marker, s=other_marker_size,
                   c=other_marker_color,label='PF Points')

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_zlabel(z_label, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.tight_layout()
    if viewpoint is not None:
        ax.view_init(elev=viewpoint[0], azim=viewpoint[1])

    if save_path is not None:
        plt.savefig(save_path+ f"/{title.replace(' ','_')}.pdf")

    plt.show()


def plot_multiple_datasets_dual_yaxis(data_list, data_names, x_name, xlim,left_y_name, right_y_name,
                                      left_y_lims = None, 
                                      right_y_lims = None,
                                      left_colors = ['tab:blue', 'tab:orange', 'tab:green'], 
                                      right_colors = ['tab:cyan', 'tab:purple', 'tab:pink'], 
                                      title=None, 
                                      xlabel=None,
                                      left_ylabel=None, 
                                      right_ylabel=None, 
                                      fontsize=12, save_path=None):
    # 创建一个新的图表
    fig, ax1 = plt.subplots(figsize=(10, 6))
     # 存储用于自定义线条样式图例的对象
    custom_legend_items = []
    # 绘制多个数据集（左侧y轴）
    for i, (data, name, color) in enumerate(zip(data_list, data_names, left_colors)):
        x = data[x_name]
        y = data[left_y_name]

        ax1.plot(x, y, color=color, label=name)
        

        ax1.set_xlabel(xlabel if xlabel else x_name)
        ax1.set_ylabel(left_ylabel if left_ylabel else left_y_name)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

    type_name = left_ylabel if left_ylabel else left_y_name
    custom_legend_items.append(Line2D([0], [0], color=color, lw=2, label=f'{type_name}'))
    # 创建右侧y轴
    ax2 = ax1.twinx()

    # 绘制多个数据集（右侧y轴）
    for i, (data, name, color) in enumerate(zip(data_list, data_names, right_colors)):
        x = data[x_name]
        y = data[right_y_name]

        ax2.plot(x, y, color=color, linestyle='dashed')
        ax2.set_ylabel(right_ylabel if right_ylabel else right_y_name)
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    type_name = right_ylabel if right_ylabel else right_y_name
    custom_legend_items.append(Line2D([0], [0], color=color, linestyle='dashed', lw=2,
                                          label=f'{type_name}'))
    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2

    # 添加图例（数据名称和线条样式）
    data_legend = ax1.legend(loc='upper left')
    style_legend = plt.legend(handles=custom_legend_items, loc='upper right')

    # 合并两个图例
    ax1.add_artist(data_legend)

    # 设置标题和x轴标签
    if title:
        plt.title(title, fontsize=fontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)

    # 设置y轴标签的字体大小
    if left_ylabel:
        ax1.set_ylabel(left_ylabel, fontsize=fontsize)
    if right_ylabel:
        ax2.set_ylabel(right_ylabel, fontsize=fontsize)

    # 设置x轴限制
    ax1.set_xlim(xlim)

    # 设置左侧和右侧y轴限制
    if left_y_lims is not None:
        ax1.set_ylim(left_y_lims[0], left_y_lims[1])
    if right_y_lims is not None:
        ax2.set_ylim(right_y_lims[0], right_y_lims[1])

    # 保存图表或显示图表
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
