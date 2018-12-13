import argparse

parser = argparse.ArgumentParser('Augmented CycleGAN')
parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
parser.add_argument('--n_gen_filters', type=int, default=32, help='Number of initial filters in generators')
parser.add_argument('--n_dis_filters', type=int, default=64, help='Number of initial filters in discriminators')
parser.add_argument('--n_enc_filters', type=int, default=32, help='Number of initial filters in encoders')
parser.add_argument('--use_dropout', action='store_true', default=False, help='Whether to use dropout in conditional resblock')
parser.add_argument('--use_sigmoid', action='store_true', default=False, help='Whether to use orginal sigmoid GAN')
parser.add_argument('--use_latent_gan', action='store_true', default=False, help='Whether to use GAN on latent codes')

parser.add_argument('--bs', type=int, default=80, help='Batchsize')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--lambda_A', type=float, default=1., help='Weight for cycle loss of domain A')
parser.add_argument('--lambda_B', type=float, default=1., help='Weight for cycle loss of domain B')
parser.add_argument('--lambda_z_B', type=float, default=.025, help='Weight for cycle loss of latent of B')
parser.add_argument('--max_norm', type=float, default=500., help='Maximum gradient norm')
parser.add_argument('--beta1', type=float, default=.5, help='Momentum coefficient')
parser.add_argument('--n_epochs', type=int, default=25, help='Number of training epochs without lr decay')
parser.add_argument('--n_epochs_decay', type=int, default=25, help='Number of training epochs with lr decay')
parser.add_argument('--print_freq', type=int, default=200, help='Logging frequency')
parser.add_argument('--valid_freq', type=int, default=600, help='Validation frequency')
parser.add_argument('--n_multi', type=int, default=10, help='Number of noise samples to generate multiple images given one image')
parser.add_argument('--n_imgs_to_save', type=int, default=20, help='Number of images to save in each iteration')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to be used')

parser.add_argument('--param_file_version', type=int, default=0, help='Weight file version to use to testing')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import neuralnet as nn
from theano import tensor as T
import numpy as np

from networks import AugmentedCycleGAN
from data_loader import Edges2Shoes, image_size

latent_dim = args.latent_dim
n_gen_filters = args.n_gen_filters
n_dis_filters = args.n_dis_filters
n_enc_filters = args.n_enc_filters
use_dropout = args.use_dropout
use_sigmoid = args.use_sigmoid
use_latent_gan = args.use_latent_gan

bs = args.bs
lr = args.lr
lambda_A = args.lambda_A
lambda_B = args.lambda_B
lambda_z_B = args.lambda_z_B
max_norm = args.max_norm
beta1 = args.beta1
n_epochs = args.n_epochs
n_epochs_decay = args.n_epochs_decay
print_freq = args.print_freq
valid_freq = args.valid_freq
n_multi = args.n_multi
n_imgs_to_save = args.n_imgs_to_save

# for testing
param_file_version = args.param_file_version


def unnormalize(x):
    return x / 2. + .5


def pre_process(x):
    downsample = nn.DownsamplingLayer((None, 3, image_size * 4, image_size * 4), 4)
    return downsample(x.dimshuffle(0, 3, 1, 2)) / 255. * 2. - 1.


def train():
    X_A_full = T.tensor4('A')
    X_B_full = T.tensor4('B')
    X_A = pre_process(X_A_full)
    X_B = pre_process(X_B_full)
    z = nn.utils.srng.normal((bs, latent_dim))
    idx = T.scalar('iter')

    X_A_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='A_plhd')
    X_B_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='B_plhd')
    lr_ = nn.placeholder(value=lr, name='lr_plhd')

    net = AugmentedCycleGAN((None, 3, image_size, image_size), latent_dim, n_gen_filters, n_dis_filters, n_enc_filters, 3,
                            use_dropout, use_sigmoid, use_latent_gan)

    nn.set_training_on()
    updates_dis, updates_gen, dis_losses, dis_preds, gen_losses, grad_norms = net.learn(X_A, X_B, z, lambda_A, lambda_B,
                                                                                        lambda_z_B, lr=lr_, beta1=beta1,
                                                                                        max_norm=max_norm)
    train_dis = nn.function([], list(dis_losses.values()), updates=updates_dis, givens={X_A_full: X_A_, X_B_full: X_B_},
                            name='train discriminators')
    train_gen = nn.function([], list(gen_losses.values()), updates=updates_gen, givens={X_A_full: X_A_, X_B_full: X_B_},
                            name='train generators')
    discriminate = nn.function([], list(dis_preds.values()), givens={X_A_full: X_A_, X_B_full: X_B_}, name='discriminate')
    compute_grad_norms = nn.function([], list(grad_norms.values()), givens={X_A_full: X_A_, X_B_full: X_B_},
                                     name='compute grad norms')

    nn.anneal_learning_rate(lr_, idx, 'linear', num_iters=n_epochs_decay)
    train_dis_decay = nn.function([idx], list(dis_losses.values()), updates=updates_dis, givens={X_A_full: X_A_, X_B_full: X_B_},
                                  name='train discriminators with decay')

    nn.set_training_off()
    fixed_z = T.constant(np.random.normal(size=(bs, latent_dim)), dtype='float32')
    fixed_multi_z = T.constant(np.repeat(np.random.normal(size=(n_multi, latent_dim)), bs, 0), dtype='float32')
    visuals = net.generate_cycle(X_A, X_B, fixed_z)
    multi_fake_B = net.generate_multi(X_A, fixed_multi_z)
    visualize_single = nn.function([], list(visuals.values()), givens={X_A_full: X_A_, X_B_full: X_B_}, name='visualize single')
    visualize_multi = nn.function([], multi_fake_B, givens={X_A_full: X_A_}, name='visualize multi')

    train_data = Edges2Shoes((X_A_, X_B_), bs, n_epochs + n_epochs_decay + 1, 'train', True)
    val_data = Edges2Shoes((X_A_, X_B_), bs, 1, 'val', False, num_data=bs)
    mon = nn.Monitor(model_name='Augmented_CycleGAN', print_freq=print_freq)

    print('Training...')
    for it in train_data:
        epoch = 1 + it // (len(train_data) // bs)

        with mon:
            res_dis = train_dis() if epoch <= n_epochs else train_dis_decay(epoch - n_epochs)
            res_gen = train_gen()
            preds = discriminate()
            grads_ = compute_grad_norms()

            mon.plot('lr', lr_.get_value())

            for j, k in enumerate(dis_losses.keys()):
                mon.plot(k, res_dis[j])

            for j, k in enumerate(gen_losses.keys()):
                mon.plot(k, res_gen[j])

            for j, k in enumerate(dis_preds.keys()):
                mon.hist(k, preds[j])

            for j, k in enumerate(grad_norms.keys()):
                mon.plot(k, grads_[j])

            if it % valid_freq == 0:
                for _ in val_data:
                    vis_single = visualize_single()
                    vis_multi = visualize_multi()

                for j, k in enumerate(visuals.keys()):
                    mon.imwrite(k, vis_single[j][:n_imgs_to_save], callback=unnormalize)

                for j, fake_B in enumerate(vis_multi):
                    mon.imwrite('fake_B_multi_%d.jpg' % j, fake_B, callback=unnormalize)

                mon.dump(nn.utils.shared2numpy(net.netG_A_B.params), 'gen_A_B.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netG_B_A.params), 'gen_B_A.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netD_A.params), 'dis_A.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netD_B.params), 'dis_B.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netE_B.params), 'enc_B.npy', 5)
                if use_latent_gan:
                    mon.dump(nn.utils.shared2numpy(net.netD_z_B.params), 'dis_z_B.npy', 5)

    mon.flush()
    mon.dump(nn.utils.shared2numpy(net.netG_A_B.params), 'gen_A_B.npy')
    mon.dump(nn.utils.shared2numpy(net.netG_B_A.params), 'gen_B_A.npy')
    mon.dump(nn.utils.shared2numpy(net.netD_A.params), 'dis_A.npy')
    mon.dump(nn.utils.shared2numpy(net.netD_B.params), 'dis_B.npy')
    mon.dump(nn.utils.shared2numpy(net.netE_B.params), 'enc_B.npy')
    if use_latent_gan:
        mon.dump(nn.utils.shared2numpy(net.netD_z_B.params), 'dis_z_B.npy')
    print('Training finished!')


if __name__ == '__main__':
    train()
