import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import neuralnet as nn
from theano import tensor as T
import numpy as np

from networks import AugmentedCycleGAN
from data_loader import Edges2Shoes, image_size

n_latent = 16
n_gen_filters = 32
n_dis_filters = 64
n_enc_filters = 32
use_dropout = False
use_sigmoid = False
n_multi = 10

bs = 80
lr = 2e-4
lambda_A = 1.
lambda_B = 1.
lambda_z_B = .025
max_norm = 500.
beta1 = .5
n_epochs = 25
n_epocs_decay = 25
print_freq = 200
valid_freq = 600
num_imgs_to_save = 20


def unnormalize(x):
    return x / 2. + .5


def pre_process(x):
    downsample = nn.DownsamplingLayer((None, 3, image_size*4, image_size*4), 4)
    return downsample(x.dimshuffle(0, 3, 1, 2)) / 255. * 2. - 1.


def train():
    X_A_full = T.tensor4('A')
    X_B_full = T.tensor4('B')
    X_A = pre_process(X_A_full)
    X_B = pre_process(X_B_full)
    z = nn.utils.srng.normal((bs, n_latent))
    idx = T.scalar('iter')

    X_A_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='A_plhd')
    X_B_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='B_plhd')
    lr_ = nn.placeholder(value=lr, name='lr_plhd')

    net = AugmentedCycleGAN((None, 3, image_size, image_size), n_latent, n_gen_filters, n_dis_filters, n_enc_filters, 3,
                            use_dropout, use_sigmoid)

    nn.set_training_on()
    updates_dis, updates_gen, dis_losses, gen_losses, _ = net.learn(X_A, X_B, z, lambda_A, lambda_B, lambda_z_B,
                                                                    lr=lr_, beta1=beta1, max_norm=max_norm)
    train_dis = nn.function([], list(dis_losses.values()), updates=updates_dis, givens={X_A_full: X_A_, X_B_full: X_B_},
                            name='train discriminators')
    train_gen = nn.function([], list(gen_losses.values()), updates=updates_gen, givens={X_A_full: X_A_, X_B_full: X_B_},
                            name='train generators')

    nn.anneal_learning_rate(lr_, idx, 'linear', num_iters=n_epocs_decay)
    train_dis_decay = nn.function([idx], list(dis_losses.values()), updates=updates_dis, givens={X_A_full: X_A_, X_B_full: X_B_},
                                  name='train discriminators with decay')

    nn.set_training_off()
    fixed_z = T.constant(np.random.normal(size=(bs, n_latent)), dtype='float32')
    fixed_multi_z = T.constant(np.repeat(np.random.normal(size=(n_multi, n_latent)), bs, 0), dtype='float32')
    visuals = net.generate_cycle(X_A, X_B, fixed_z)
    multi_fake_B = net.generate_multi(X_A, fixed_multi_z)
    visualize_single = nn.function([], list(visuals.values()), givens={X_A_full: X_A_, X_B_full: X_B_}, name='visualize single')
    visualize_multi = nn.function([], multi_fake_B, givens={X_A_full: X_A_}, name='visualize multi')

    train_data = Edges2Shoes((X_A_, X_B_), bs, n_epochs+n_epocs_decay+1, 'train', True)
    val_data = Edges2Shoes((X_A_, X_B_), bs, 1, 'val', False, num_data=bs)
    mon = nn.Monitor(model_name='Augmented_CycleGAN', print_freq=print_freq)

    print('Training...')
    for it in train_data:
        epoch = 1 + it // (len(train_data) // bs)

        with mon:
            res_dis = train_dis() if epoch <= n_epochs else train_dis_decay(epoch - n_epochs)
            res_gen = train_gen()

            mon.plot('lr', lr_.get_value())

            for j, k in enumerate(dis_losses.keys()):
                mon.plot(k, res_dis[j])

            for j, k in enumerate(gen_losses.keys()):
                mon.plot(k, res_gen[j])

            if it % valid_freq == 0:
                for _ in val_data:
                    vis_single = visualize_single()
                    vis_multi = visualize_multi()

                for j, k in enumerate(visuals.keys()):
                    mon.imwrite(k, vis_single[j][:num_imgs_to_save], callback=unnormalize)

                for j, fake_B in enumerate(vis_multi):
                    mon.imwrite('fake_B_multi_%d.jpg' % j, fake_B, callback=unnormalize)

                mon.dump(nn.utils.shared2numpy(net.netG_A_B.params), 'gen_A_B.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netG_B_A.params), 'gen_B_A.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netD_A.params), 'dis_A.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netD_B.params), 'dis_B.npy', 5)
                mon.dump(nn.utils.shared2numpy(net.netE_B.params), 'enc_B.npy', 5)

    mon.flush()
    mon.dump(nn.utils.shared2numpy(net.netG_A_B.params), 'gen_A_B.npy')
    mon.dump(nn.utils.shared2numpy(net.netG_B_A.params), 'gen_B_A.npy')
    mon.dump(nn.utils.shared2numpy(net.netD_A.params), 'dis_A.npy')
    mon.dump(nn.utils.shared2numpy(net.netD_B.params), 'dis_B.npy')
    mon.dump(nn.utils.shared2numpy(net.netE_B.params), 'enc_B.npy')
    print('Training finished!')


if __name__ == '__main__':
    train()
