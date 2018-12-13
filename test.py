from train import *


def test():
    X_A_full = T.tensor4('A')
    X_B_full = T.tensor4('B')
    X_A = pre_process(X_A_full)
    X_B = pre_process(X_B_full)

    X_A_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='A_plhd')
    X_B_ = nn.placeholder((bs, 3, image_size*4, image_size*4), name='B_plhd')

    net = AugmentedCycleGAN((None, 3, image_size, image_size), latent_dim, n_gen_filters, n_dis_filters, n_enc_filters, 3,
                            use_dropout, use_sigmoid, use_latent_gan)

    nn.set_training_off()
    fixed_z = T.constant(np.random.normal(size=(bs, latent_dim)), dtype='float32')
    fixed_multi_z = T.constant(np.repeat(np.random.normal(size=(n_multi, latent_dim)), bs, 0), dtype='float32')
    visuals = net.generate_cycle(X_A, X_B, fixed_z)
    multi_fake_B = net.generate_multi(X_A, fixed_multi_z)
    visualize_single = nn.function([], list(visuals.values()), givens={X_A_full: X_A_, X_B_full: X_B_},
                                   name='test single')
    visualize_multi = nn.function([], multi_fake_B, givens={X_A_full: X_A_}, name='test multi')

    val_data = Edges2Shoes((X_A_, X_B_), bs, 1, 'val', False)
    mon = nn.Monitor(current_folder='results/Augmented_CycleGAN/run1')
    nn.utils.numpy2shared(mon.load('gen_A_B-%d.npy' % param_file_version), net.netG_A_B.params)
    nn.utils.numpy2shared(mon.load('gen_B_A-%d.npy' % param_file_version), net.netG_B_A.params)
    nn.utils.numpy2shared(mon.load('dis_A-%d.npy' % param_file_version), net.netD_A.params)
    nn.utils.numpy2shared(mon.load('dis_B-%d.npy' % param_file_version), net.netD_B.params)
    nn.utils.numpy2shared(mon.load('enc_B-%d.npy' % param_file_version), net.netE_B.params)
    if use_latent_gan:
        nn.utils.numpy2shared(mon.load('dis_z_B.npy', version=param_file_version), net.netD_z_B.params)

    print('Testing...')
    for _ in val_data:
        vis_single = visualize_single()
        vis_multi = visualize_multi()

        for j, k in enumerate(visuals.keys()):
            mon.imwrite('test_' + k, vis_single[j][:n_imgs_to_save], callback=unnormalize)

        for j, fake_B in enumerate(vis_multi):
            mon.imwrite('test_fake_B_multi_%d.jpg' % j, fake_B, callback=unnormalize)
    mon.flush()
    print('Testing finished!')


if __name__ == '__main__':
    test()
