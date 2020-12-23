import math
import os
import torch
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt

# import normalizing flows
from flow import NormalizingFlow

# import psi-network
from psi_network import ReLUNN
from psi_network import TanhNN

# import loss functions
from losses import Relative_Entropy
from losses import JKO_loss
from losses import Wass_loss
from losses import JKO_loss_modified
from losses import Wass_loss_modified

# import potential functions
from Quadratic_Function import Quadratic_Function_torch
from Quadratic_Function2 import Quadratic_Function2_torch
from Quadratic_Function_high_dimension import Quadratic_Function_high_dim_torch
from Rosenbrock_Function import Rosenbrock_Function_torch
from S_Tang_Function import S_Tang_Function_torch

# import other utilities
from utils import random_normal_samples
from plot_sample import plot_sample
from record_sample import record_sample_coord
from plot_density import plot_density, plot_contour_density, plot_3d_density
from utils import create_nodes, psi_nodes


############################################################################################
#  Implementation of Algorithm 4.1 of "NEURAL PARAMETRIC FOKKER-PLANCK EQUATION"           #
############################################################################################


#########################
# Setting Parameters    #
#########################

# select random seed
torch.manual_seed(42)

# dimension of the problem
dimension = 2  # 10  # 30 

# length of normalizing flow
flowlength = 40  # for higher dimensions 10, 30, 50, please try 60, 60, 100

# configuration of psi-network
psi_network_length = 6  # for higher dimensions 10, 30, 50, please try 15, 30, 30
hidden_dim = 20

# time stepsize
h = 0.005

# number of time steps
T = 800  # 600

# sample size for evaluating relative entropy (KL-divergence)
KL_sample_size = 12000

# parameters for outer iterations
theta_iterations = 30  # 50
theta_batchsize = 5000
theta_lr = 0.0005
theta_opt_momentum = 0.9
lrdecay = 0.999

# set up epsilon (c.f. Remark 4.9), we may directly choose epsilon = theta_lr
epsilon = 0.0005  # theta_lr

# parameters for inner iterations
psi_iterations = 60  # 100
psi_batchsize = 1000
psi_lr = 0.005
psiopt_momentum = 0.8
psi_lrdecay = 0.99

# how often do we record data
record_KL_period = 1  # record KL losses
record_outer_loss_period = 10  # plot outer loss curve
record_samples_period = 10  # plot samples and estimated densities
record_psi_period = 15  # 50   # record inner loss curves and plot graphs of psi
plot_num = 6000  # number of sample points to plot

# plot on a-b-plane:
a = 0 # 5
b = 1 # 15
# plotting range (on square [-L,L]x[-L,L])
L = 7
# nbins used for generate heat map that corresponds to the samples
nbins = 600


###################################################
# Setting up functions and optimizers.            #
###################################################

# set up normalizing flow and an auxiliary flow
flow = NormalizingFlow(dim=dimension, flow_length=flowlength)
flow_auxil = NormalizingFlow(dim=dimension, flow_length=flowlength)  # auxiliary normalizing flow

# set up psi-flow (Fully connected neural network with ReLU, Tanh or other kinds of activations)
psi = ReLUNN(network_length=psi_network_length, hidden_dimension=hidden_dim, input_dimension=dimension, output_dimension=1)
# psi = TanhNN(network_length=psi_network_length, hidden_dimension=hidden_dim, input_dimension=dimension, output_dimension=1)

# define potential function
potential = S_Tang_Function_torch
# potential = Quadratic_Function_high_dim_torch
# potential = Quadratic_Function_torch
# potential = Quadratic_Function2_torch
# potential = Rosenbrock_Function_torch

# define loss functions
relative_entropy_loss = Relative_Entropy(potential=potential, dimension=dimension)
outer_loss = JKO_loss_modified(potential=potential, dimension=dimension, epsilon=epsilon, h=h)
inner_loss = Wass_loss_modified(epsilon=epsilon)

# define outer optimizer, we use ADAM here
theta_optimizer = optim.Adam(flow.parameters(), lr=theta_lr, betas=(0.9, 0.999), eps=1e-08)
scheduler = optim.lr_scheduler.ExponentialLR(theta_optimizer, lrdecay)

# define inner optimizer, we use ADAM here
psi_optimizer = optim.Adam(psi.parameters(), lr=psi_lr, betas=(0.9, 0.999), eps=1e-08)
psi_scheduler = optim.lr_scheduler.ExponentialLR(psi_optimizer, psi_lrdecay)

# Create a new folder
folder = os.getcwd() + '/ Styblinski-Tang Potential DIM = {}'.format(dimension)
os.makedirs(folder)

# Set up path
save_path = os.getcwd() + '/ Styblinski-Tang Potential DIM = {}'.format(dimension)

# use to record KL losses along time
record_KL_loss_list = []

# Algorithm starts here
for iter in range(1, T + 1):

    # copy current parameters of T_\theta to an auxiliary normalizing flow
    flow_auxil.load_state_dict(flow.state_dict())

    # compute KL loss (relative entropy) at this time step
    samples = Variable(random_normal_samples(KL_sample_size, dim=dimension))
    zk, log_jacobians = flow(samples)
    KL_loss = relative_entropy_loss(zk, log_jacobians)

    # initialize the parameters of psi
    psi.initialization()

    # print outer current time step
    print(iter)

    # use to record outer loss (loss_jko)
    outer_loss_list = []

    # outer iteration starts
    for outer_iteration in range(1, theta_iterations + 1):

        print("finish one outer iteration")

        inner_loss_list = []
        # inner iteration starts
        for inner_iteration in range(1, psi_iterations + 1):

            # conduct inner optimization
            # psi_scheduler.step()
            samples = Variable(random_normal_samples(psi_batchsize, dim=dimension), requires_grad=True)
            original_transformed_samples, _ = flow_auxil(samples)
            v_original = Variable(original_transformed_samples, requires_grad=True)
            current_transformed_samples, _ = flow(samples)
            v_current = Variable(current_transformed_samples)
            uk = psi(v_original)

            gradients = torch.autograd.grad(outputs=uk, inputs=v_original, grad_outputs=torch.ones(uk.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]

            psi_optimizer.zero_grad()
            psi_loss = inner_loss(v_original, v_current, gradients)
            psi_loss.backward()
            psi_optimizer.step()

            # record inner loss
            inner_loss_list.append(psi_loss.item())

        # conduct outer optimization
        samples_1 = Variable(random_normal_samples(theta_batchsize, dim=dimension))
        xxk, _ = flow_auxil(samples_1)
        xk = Variable(xxk, requires_grad=True)
        zk, log_jacobians = flow(samples_1)
        psik = psi(xk)
        gradients = \
        torch.autograd.grad(outputs=psik, inputs=xk, grad_outputs=torch.ones(psik.size()), create_graph=True,
                            retain_graph=True, only_inputs=True)[0]
        theta_optimizer.zero_grad()
        theta_loss = outer_loss(zk, log_jacobians, gradients)
        theta_loss.backward()
        theta_optimizer.step()

        # record the outer loss
        outer_loss_list.append(theta_loss.item())

        # record inner losses, plot inner loss curve and plot graphs of psi functions
        if iter % record_psi_period == 0:

            # write inner loss in a text file
            filename_psilosslist = os.path.join(save_path, "psi_loss: {}-th time step, {}-th outer iteration".format(iter, outer_iteration) + ".txt")
            f = open(filename_psilosslist, "w+")
            k = 0
            for innerloss_data in inner_loss_list:
                k = k + 1
                f.write("{}. inner_loss: {} \n".format(k, innerloss_data))

            # plot inner loss curve
            num = psi_iterations
            Nodes = create_nodes(num, 1)
            Record_loss = torch.tensor(inner_loss_list)
            plt.scatter(Nodes, Record_loss)
            filename_psiloss = os.path.join(save_path, "inner loss: {}-th time step, {}-th outer iteration".format(iter, outer_iteration))
            plt.savefig(filename_psiloss)
            plt.close()

            # plot the graphs of psi(x)
            interval_width = 0.3
            num_of_intervals = 30
            Nodes = psi_nodes(interval_width, num_of_intervals, dimension)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            NUMBER = (num_of_intervals + 1) * (num_of_intervals + 1)
            for k in range(0, NUMBER):
                x_node = Nodes[k][0]
                y_node = Nodes[k][1]
                graph_psi = psi(Nodes[k])
                ax.scatter(x_node, y_node, graph_psi.detach().numpy())
            filename_graph_psi = os.path.join(save_path, "graph of psi(x) [{}th time step, {}th outer iteration] ".format(iter, outer_iteration))
            plt.savefig(filename_graph_psi)
            plt.close()

    if iter % record_KL_period == 0:

        print("Loss on iteration {}: {}".format(iter, KL_loss.data.item()))
        record_KL_loss_list.append(KL_loss.data.item())

    if iter % record_samples_period == 0:

        # plot samples
        plot_samples = Variable(random_normal_samples(plot_num, dim=dimension))
        zk, _ = flow(plot_samples)
        plot_sample(a, b, zk.data.numpy(), iter, flowlength, L, save_path)

        # record the coordinates of samples
        record_sample_coord(zk.data.numpy(), iter, save_path)

        # plot estimated density of the samples
        plot_density(zk, a, b, nbins, iter, L, save_path)
        plot_contour_density(zk, a, b, nbins, iter, L, save_path)
        plot_3d_density(zk, a, b, nbins, iter, L, save_path)

    if iter % record_outer_loss_period == 0:

        # plot outer loss
        num = theta_iterations
        Nodes = create_nodes(num, 1)
        Record_loss = torch.tensor(outer_loss_list)
        plt.scatter(Nodes, Record_loss)
        filename_outer_loss = os.path.join(save_path, "outer loss at {}th time step".format(iter))
        plt.savefig(filename_outer_loss)
        plt.close()

# plot KL loss curve
num = math.floor(iter/record_KL_period)
Nodes = create_nodes(num, record_KL_period)
Record_loss = torch.tensor(record_KL_loss_list)
plt.scatter(Nodes, Record_loss)
filename_Loss = os.path.join(save_path,  "KL loss curve ")
plt.savefig(filename_Loss)
plt.close()

# writing down parameters & data
# write down parameters
filename1 = os.path.join(save_path, "parameters"+".txt")
f = open(filename1, "w+")
f.write("dimension of the problem is %d \n" % dimension)
f.write("\n")
f.write("flowlength = %d \n" % flowlength)
f.write("\n")
f.write("number of time steps = %d \n" % T)
f.write("stepsize h = %f \n" % h)
f.write("sample size for evaluating relative entropy (KL-divergence) = %d \n" % KL_sample_size)
f.write("outer batchsize = %d \n" % theta_batchsize)
f.write("number of outer iterations = %d \n" % theta_iterations)
f.write("outer learning rate = %f \n" % theta_lr)
f.write("\n")
f.write("inner batchsize = %d \n" % psi_batchsize)
f.write("number of inner iterations = %d \n" % psi_iterations)
f.write("inner learning rate = %f \n" % psi_lr)
f.write("inner learning rate decay = %f \n" % psi_lrdecay)
f.write("\n")
f.write("period of recording KL losses = %d \n" % record_KL_period)
f.write("period of recording samples = %d \n" % record_samples_period)
f.write("number of samples used for plotting = %d \n" % plot_num)
f.write("\n")
f.write("plot coordinate a = %d \n" % a)
f.write("plot coordinate b = %d \n" % b)
f.write("plot range L = %f \n" % L)
f.write("KDE nbins = %d \n" % nbins)
f.write("\n")
f.close()

# write down losses
filename2 = os.path.join(save_path, "KL loss data"+".txt")
f = open(filename2, "w+")
k = 0
for data in record_KL_loss_list:
    k = k + 1
    f.write("{}.loss:{} \n".format(k, data))
