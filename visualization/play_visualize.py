#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2023/08/02 13:45:38
@Author  :   Cao Zhanxiang 
@Version :   1.0
@Contact :   caozx1110@163.com
@License :   (C)Copyright 2023
@Desc    :   None
'''
import matplotlib.pyplot as plt

def plot_dof_pos(dof_pos_list, tar_dof_pos_list):
    plt.figure(figsize=(20, 15))
    
    for i in range(0, 12):
        plt.subplot(2, 2, i // 3 + 1)
        plt.plot(dof_pos_list[:, i], label=f"dof_pos_{i}")
        # plt.plot(dof_vel_list[:, i], label=f"dof_vel_{i}")
        plt.plot(tar_dof_pos_list[:, i], label=f"tar_dof_pos_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("dof's curve")
    
    plt.savefig('./data/dof_curve.png', dpi=200)
    
def plot_dof_vel(dof_vel_list):
    plt.figure(figsize=(20, 15))
    
    for i in range(0, 12):
        plt.subplot(2, 2, i // 3 + 1)
        plt.plot(dof_vel_list[:, i], label=f"dof_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("dof's vel curve")
    
    plt.savefig('./data/dof_vel_curve.png', dpi=200)
    
def plot_base(base_lin_vel_list, base_ang_vel_list, base_quat_list, foot_contact_list):
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)
    for i in range(0, 3):
        plt.plot(base_lin_vel_list[:, i], label=f"base_lin_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("base's lin vel curve")
    
    plt.subplot(2, 2, 2)
    for i in range(0, 3):
        plt.plot(base_ang_vel_list[:, i], label=f"base_ang_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("base's ang vel curve")
    
    plt.subplot(2, 2, 3)
    for i in range(0, 3):
        plt.plot(base_quat_list[:, i], label=f"base_rot_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("base's rot curve")
    
    plt.subplot(2, 2, 4)
    for i in range(0, 4):
        plt.plot(foot_contact_list[:, i], label=f"foot_contact_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("foot's contact curve")
        
    plt.savefig('./data/base_curve.png', dpi=200)
    
    
def plot_command(command_list, base_lin_vel_list, base_ang_vel_list):
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.plot(command_list[:, 0], label=f"command_x")
    print(command_list[:, 0].shape)
    plt.plot(base_lin_vel_list[:, 0], label=f"base_lin_vel_x")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/ms')
    plt.title("x's curve")
    
    plt.subplot(3, 1, 2)
    plt.plot(command_list[:, 1], label=f"command_y")
    plt.plot(base_lin_vel_list[:, 1], label=f"base_lin_vel_y")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/ms')
    plt.title("y's curve")
    
    plt.subplot(3, 1, 3)
    plt.plot(command_list[:, 2], label=f"command_z")
    plt.plot(base_ang_vel_list[:, 2], label=f"base_ang_vel_z")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/ms')
    plt.title("z's curve")
    
    plt.savefig('./data/command_curve.png', dpi=200)
    
def plot_action(action_list):
    plt.figure(figsize=(20, 15))
    
    for i in range(0, 12):
        plt.subplot(2, 2, i // 3 + 1)
        plt.plot(action_list[:, i], label=f"action_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("action's curve")
    
    plt.savefig('./data/action_curve.png', dpi=200)

def plot_dof_torque(torque_list):
    plt.figure(figsize=(20, 15))
    
    for i in range(0, 12):
        plt.subplot(2, 2, i // 3 + 1)
        plt.plot(torque_list[:, i], label=f"torque_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/ms')
        plt.title("torque's curve")
    
    plt.savefig('./data/torque_curve.png', dpi=200)

