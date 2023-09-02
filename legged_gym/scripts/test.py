import torch

def inverse_kinematics2(x, y, z):
    l0 = 0.08025
    l1 = 0.25
    l2 = 0.25

    pos_action = torch.zeros(2,18) # num_envs * num_dof(18)

    z_r = z[:,0:3] # right legs
    z_l = z[:,3:6] # left legs
    y_r = y[:,0:3] # right legs
    y_l = y[:,3:6] # left legs

    print(torch.atan2(z_r,-y_r))
    theta0_r = -torch.atan2(z_r,-y_r) - torch.atan2(torch.sqrt(y_r**2+z_r**2-l0**2),l0*torch.ones_like(z_r))
    theta0_l = torch.atan2(z_l,y_l) + torch.atan2(torch.sqrt(y_l**2+z_l**2-l0**2),l0*torch.ones_like(z_r))

    L = torch.sqrt(y**2+z**2-l0**2)

    theta1 = -torch.atan2(x,torch.sqrt(L**2-x**2)) + torch.acos((l1**2+L**2-l2**2)/2/l1/L)
    theta2 = -torch.acos(L**2-l1**2-l2**2/2/l1/l2)

    pos_action[:,[0,3,6]] = theta0_r
    pos_action[:,[9,12,15]] = theta0_l
    pos_action[:,[1,4,7,10,13,16]] = theta1
    pos_action[:,[2,5,8,11,14,17]] = theta2

    return pos_action

x = torch.tensor([[0,0.1,0,0.1,0,0.1],[0,0.1,0,0.1,0,0.1]])
y = torch.tensor([[-0.08025,-0.08025,-0.08025,0.08025,0.08025,0.08025],[-0.08025,-0.08025,-0.08025,0.08025,0.08025,0.08025]])
z = torch.tensor([[-0.36,-0.36,-0.36,-0.36,-0.36,-0.36],[-0.36,-0.36,-0.36,-0.36,-0.36,-0.36]])
pos_ation = inverse_kinematics2(x,y,z)
print(pos_ation)