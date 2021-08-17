import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from model import CNN
from train import compute_loss


def randomize_layers(model, device, rand_ndx_list=[]):
    """ Randomize model parameters for chosen layers.

    Args:
        model: model for which to randomize the parameters
        device (str): devince on which we carry out the computation. 'cpu' or 'cuda'
        rand_ndx_list (list): list of integers corresponding to layers in the model. 
            Note that the head layer is assigned the integer 1.

    Returns:
        randomized model.
    """
    name_list = ['Conv2d', 'Linear']
    ndx = 1
    randomized_model = CNN().to(device)
    randomized_model.eval()
    model = model.to(device)
    model.eval()
    for m, randomized_m in reversed(list(zip(model.children(), randomized_model.children()))):
        if m.__class__.__name__ in name_list:
            if ndx in rand_ndx_list:
                nn.init.uniform_(randomized_m.weight.data, -1.0, 1.0)
                nn.init.uniform_(randomized_m.bias.data, -1.0, 1.0)
                # print(m.__class__.__name__, 'randomize')
            else:
                randomized_m.weight.data = m.weight.data.clone()
                randomized_m.bias.data = m.bias.data.clone()
                # print(m.__class__.__name__, 'copy')
            ndx += 1
        # else:
        #     print(m.__class__.__name__, 'skip')
    return randomized_model


def saliency(model, sample_img, sample_label, device, N_smooth=0, std=0.2, rand_ndx_list=[]):
    """ Computes the saliency map using gradient / smooth gradient. 

    Args:
        model: prediction model.
        sample_img (torch.tensor): batch of images
        sample_label (torch.tensor): batch of labels corresponding to the sample_img
        device (str): device on which the computation is carried out. 'cpu' or 'cuda'
        N_smooth (int): number of saliency maps that we average over by adding Gaussian noise of zero mean.
            When N_smooth = 0, no noise is added.
        std (float): standard devication of Gaussian noise.
        rand_ndx_list (list): list of integers corresponding to the layers that will be randomized.
            See randomize_layers

    Returns:
        np.array: batch of saliency map for the sample_img.
    """
    randomized_model = randomize_layers(model, device, rand_ndx_list)
    sample_img = sample_img.to(device)
    sample_label = sample_label.to(device)
    assert type(N_smooth) == int and N_smooth >= 0

    img_iter = torch.zeros(sample_img.shape, requires_grad=True, device=device)
    if N_smooth == 0:
        img_iter.data = sample_img.data.clone()
        loss, TP = compute_loss(randomized_model, img_iter, sample_label)
        loss.backward()
        return img_iter.grad.cpu().numpy()
    else:
        for i in range(N_smooth):
            img_iter.data = sample_img.data + std * torch.randn(sample_img.shape, device=device)
            loss, TP = compute_loss(randomized_model, img_iter, sample_label)
            loss.backward()
        return img_iter.grad.cpu().numpy()


def plot_saliency(num_plots, img_grad, img_grad_sm, sample_img, fig_name='', cmap='gray'):
    """ Plots saliency map.
    Args:
        num_plots (int): number of images for which we compute the saliency map
            This number should be less than the number of batches in img_grad
        img_grad (np.array): batch saliency maps computed using gradient
        img_grad_sm (np.array): batch saliency maps computed using smooth gradient
        sample_img (np.array/torch.tensor): batch of sample img
        fig_name (str): name figure to be saved.

    Returns:
        None

    """
    fig, ax = plt.subplots(num_plots, 3, figsize=(6, 2 * num_plots))
    for plt_idx in range(num_plots):
        ax[plt_idx, 0].imshow(img_grad[plt_idx, 0, :, :], norm=colors.Normalize(), cmap=cmap)
        ax[plt_idx, 1].imshow(img_grad_sm[plt_idx, 0, :, :], norm=colors.Normalize(), cmap=cmap)
        ax[plt_idx, 2].imshow(sample_img[plt_idx, 0, :, :].data.numpy(), norm=colors.Normalize(), cmap=cmap)
        for i in range(3):
            ax[plt_idx, i].axes.xaxis.set_ticks([])
            ax[plt_idx, i].axes.yaxis.set_ticks([])
    ax[num_plots - 1, 0].set_xlabel('Gradient')
    ax[num_plots - 1, 1].set_xlabel('Smooth \n Gradient')
    ax[num_plots - 1, 2].set_xlabel('Original \n Image')
    plt.tight_layout()
    plt.show()
    if fig_name:
        fig.savefig(fig_name, format='png')
