import neuralnet as nn
from theano import tensor as T
from functools import partial
from collections import OrderedDict


def cin_resnet_block(input_shape, z_dim, padding, norm_layer, use_dropout, use_bias, block_name, **kwargs):
    num_filters = input_shape[1]

    block = nn.Sequential(input_shape=input_shape, layer_name=block_name)
    block.append(nn.Conv2DLayer(input_shape, num_filters, 3, border_mode=padding, no_bias=not use_bias, activation=None,
                                layer_name=block_name + '/conv1'))
    block.append(norm_layer(input_shape, z_dim, layer_name=block_name + '/CIN'))
    block.append(nn.ActivationLayer(block.output_shape, layer_name=block_name + '/relu1'))
    if use_dropout:
        block.append(nn.DropoutLayer(block.output_shape, .5, layer_name=block_name + '/dropout'))

    block.append(nn.Conv2DLayer(block.output_shape, num_filters, 3, border_mode=padding, no_bias=not use_bias,
                                activation=None, layer_name=block_name + '/conv2'))
    block.append(nn.InstanceNormLayer(block.output_shape, block_name + '/IN'))
    return block


def resnet_block(input_shape, padding, norm_layer, use_dropout, use_bias, block_name, **kwargs):
    num_filters = input_shape[1]

    block = nn.Sequential(input_shape=input_shape, layer_name=block_name)
    block.append(nn.Conv2DLayer(input_shape, num_filters, 3, border_mode=padding, no_bias=not use_bias, activation='relu',
                                layer_name=block_name + '/conv1'))
    if use_dropout:
        block.append(nn.DropoutLayer(block.output_shape, .5, layer_name=block_name + '/dropout'))

    block.append(nn.Conv2DLayer(block.output_shape, num_filters, 3, border_mode=padding, no_bias=not use_bias,
                                activation=None, layer_name=block_name + '/conv2'))
    block.append(norm_layer(block.output_shape, block_name + '/IN'))
    return block


class ResnetBlock(nn.ResNetBlock):
    def __init__(self, input_shape, padding, norm_layer, use_dropout, use_bias, layer_name='Resnet Block'):
        super(ResnetBlock, self).__init__(input_shape, padding=padding, norm_layer=norm_layer, use_dropout=use_dropout,
                                          use_bias=use_bias, num_filters=input_shape[1], stride=(1, 1),
                                          activation='relu', normalization=None, block=resnet_block,
                                          layer_name=layer_name)
        self.padding = padding
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.use_bias = use_bias


class CINResnetBlock(nn.ResNetBlock):
    def __init__(self, input_shape, noise_dim, padding, norm_layer, use_dropout, use_bias, layer_name='CIN Resnet Block'):
        super(CINResnetBlock, self).__init__(input_shape, z_dim=noise_dim, padding=padding, norm_layer=norm_layer,
                                             use_dropout=use_dropout, use_bias=use_bias, num_filters=input_shape[1],
                                             stride=(1, 1), activation='relu', normalization=None, block=cin_resnet_block,
                                             layer_name=layer_name)
        self.noise_dim = noise_dim
        self.padding = padding
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.use_bias = use_bias


class CINResnetGen(nn.Sequential):
    def __init__(self, input_shape, num_filters, n_latent, output_dim, norm_layer=nn.ConditionalInstanceNorm2DLayer,
                 use_dropout=False, padding='ref', name='CIN Resnet Generator'):
        super(CINResnetGen, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.Conv2DLayer(input_shape, num_filters, 7, border_mode='ref', activation=None, no_bias=False,
                                   layer_name=name+'/conv1'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name+'/cin1'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu1'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters*2, 3, border_mode='half', activation=None, no_bias=False,
                                   layer_name=name+'/conv2'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name+'/cin2'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu2'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters*4, 3, stride=2, border_mode='half', activation=None,
                                   no_bias=False, layer_name=name+'/conv3'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name+'/cin3'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu3'))

        for i in range(3):
            self.append(CINResnetBlock(self.output_shape, n_latent, padding, norm_layer, use_dropout, True,
                                       name + '/CIN ResBlock %d' % (i + 1)))

        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, 2 * num_filters, 3, stride=(2, 2), padding='half',
                                            activation=None, layer_name=name + '/deconv'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name+'/cin4'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name+'/relu4'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters, 3, border_mode='half', activation=None, no_bias=False,
                                   layer_name=name+'/conv5'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name+'/cin5'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu5'))

        self.append(nn.Conv2DLayer(self.output_shape, output_dim, 7, activation=None, layer_name=name+'/output'))
        self.append(nn.ActivationLayer(self.output_shape, 'tanh', layer_name=name+'/output_act'))


class ResnetGen(nn.Sequential):
    def __init__(self, input_shape, num_filters, output_dim, norm_layer=partial(nn.InstanceNormLayer, activation=None),
                 use_dropout=False, padding='ref', name='Resnet Generator'):
        super(ResnetGen, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.Conv2DLayer(input_shape, num_filters, 7, border_mode='ref', activation=None, no_bias=False,
                                   layer_name=name+'/conv1'))
        self.append(norm_layer(self.output_shape, layer_name=name+'/cin1'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu1'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters*2, 3, border_mode='half', activation=None, no_bias=False,
                                   layer_name=name+'/conv2'))
        self.append(norm_layer(self.output_shape, layer_name=name+'/cin2'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu2'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters*4, 3, stride=2, border_mode='half', activation=None,
                                   no_bias=False, layer_name=name+'/conv3'))
        self.append(norm_layer(self.output_shape, layer_name=name+'/cin3'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu3'))

        for i in range(3):
            self.append(ResnetBlock(self.output_shape, padding, norm_layer, use_dropout, True,
                                    name + '/ResBlock %d' % (i + 1)))

        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, 2 * num_filters, 3, stride=(2, 2), padding='half',
                                            activation=None, layer_name=name + '/deconv'))
        self.append(norm_layer(self.output_shape, layer_name=name+'/cin4'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', name+'/relu4'))

        self.append(nn.Conv2DLayer(self.output_shape, num_filters, 3, border_mode='half', activation=None, no_bias=False,
                                   layer_name=name+'/conv5'))
        self.append(norm_layer(self.output_shape, layer_name=name+'/cin5'))
        self.append(nn.ActivationLayer(self.output_shape, 'relu', layer_name=name+'/relu5'))

        self.append(nn.Conv2DLayer(self.output_shape, output_dim, 7, activation=None, layer_name=name+'/output'))
        self.append(nn.ActivationLayer(self.output_shape, 'tanh', layer_name=name+'/output_act'))


class CINDiscriminator(nn.Sequential):
    def __init__(self, input_shape, n_latent, num_filters=64, norm_layer=nn.ConditionalInstanceNorm2DLayer,
                 use_sigmoid=False, use_bias=True, name='CIN Discriminator'):
        super(CINDiscriminator, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.Conv2DLayer(self.output_shape, num_filters, 4, stride=2, border_mode=1, no_bias=not use_bias,
                                   activation='lrelu', alpha=.2, layer_name=name + '/conv1'))

        self.append(nn.Conv2DLayer(self.output_shape, 2 * num_filters, 4, stride=2, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv2'))
        self.append(norm_layer(self.output_shape, n_latent, name + '/bn2'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act2'))

        self.append(nn.Conv2DLayer(self.output_shape, 4 * num_filters, 4, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv3'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name + '/bn3'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act3'))

        self.append(nn.Conv2DLayer(self.output_shape, 5 * num_filters, 4, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv4'))
        self.append(norm_layer(self.output_shape, n_latent, layer_name=name + '/bn4'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act4'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 1, 4, border_mode=1, layer_name=name + '/output', activation=None))

        if use_sigmoid:
            self.append(nn.ActivationLayer(self.output_shape, 'sigmoid', name + '/sigmoid'))


class Discriminator(nn.Sequential):
    def __init__(self, input_shape, num_filters=64, norm_layer=partial(nn.BatchNormLayer, activation=None),
                 use_sigmoid=False, use_bias=True, name='Discriminator'):
        super(Discriminator, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.Conv2DLayer(self.output_shape, num_filters, 4, stride=2, border_mode=1, no_bias=not use_bias,
                                   activation='lrelu', alpha=.2, layer_name=name + '/conv1'))

        self.append(nn.Conv2DLayer(self.output_shape, 2 * num_filters, 4, stride=2, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv2'))
        self.append(norm_layer(self.output_shape, name + '/bn2'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act2'))

        self.append(nn.Conv2DLayer(self.output_shape, 4 * num_filters, 4, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv3'))
        self.append(norm_layer(self.output_shape, layer_name=name + '/bn3'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act3'))

        self.append(nn.Conv2DLayer(self.output_shape, 4 * num_filters, 4, border_mode=1, no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv4'))
        self.append(norm_layer(self.output_shape, layer_name=name + '/bn4'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act4'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 1, 4, border_mode=1, layer_name=name + '/output', activation=None))

        if use_sigmoid:
            self.append(nn.ActivationLayer(self.output_shape, 'sigmoid', name + '/sigmoid'))


class DiscriminatorEdges(nn.Sequential):
    def __init__(self, input_shape, num_filters=64, norm_layer=partial(nn.BatchNormLayer, activation=None),
                 use_sigmoid=False, use_bias=True, name='Discriminator Edges'):
        super(DiscriminatorEdges, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.Conv2DLayer(self.output_shape, num_filters, 3, stride=2, border_mode='half', no_bias=not use_bias,
                                   activation='lrelu', alpha=.2, layer_name=name + '/conv1'))

        self.append(nn.Conv2DLayer(self.output_shape, 2 * num_filters, 3, stride=2, border_mode='half', no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv2'))
        self.append(norm_layer(self.output_shape, name + '/bn2'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act2'))

        self.append(nn.Conv2DLayer(self.output_shape, 4 * num_filters, 3, border_mode='half', no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv3'))
        self.append(norm_layer(self.output_shape, layer_name=name + '/bn3'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act3'))

        self.append(nn.Conv2DLayer(self.output_shape, 4 * num_filters, 3, border_mode='half', no_bias=not use_bias,
                                   activation=None, layer_name=name + '/conv4'))
        self.append(norm_layer(self.output_shape, layer_name=name + '/bn4'))
        self.append(nn.ActivationLayer(self.output_shape, activation='lrelu', alpha=.2, layer_name=name + '/act4'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 1, 4, border_mode='valid', layer_name=name + '/output', activation=None))

        if use_sigmoid:
            self.append(nn.ActivationLayer(self.output_shape, 'sigmoid', name + '/sigmoid'))


class DiscriminatorLatent(nn.Sequential):
    def __init__(self, input_shape, n_nodes, use_sigmoid=False, name='Latent Discriminator'):
        super(DiscriminatorLatent, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.FCLayer(self.output_shape, n_nodes, activation=None, layer_name=name+'/fc1'))
        self.append(nn.BatchNormLayer(self.output_shape, name+'/bn1', activation='lrelu', alpha=.2))

        self.append(nn.FCLayer(self.output_shape, n_nodes, activation=None, layer_name=name+'/fc2'))
        self.append(nn.BatchNormLayer(self.output_shape, name+'/bn2', activation='lrelu', alpha=.2))

        self.append(nn.FCLayer(self.output_shape, n_nodes, activation=None, layer_name=name+'/fc3'))
        self.append(nn.BatchNormLayer(self.output_shape, name+'/bn3', activation='lrelu', alpha=.2))

        self.append(nn.FCLayer(self.output_shape, 1, activation=None, layer_name=name+'/output'))

        if use_sigmoid:
            self.append(nn.ActivationLayer(self.output_shape, 'sigmoid', name+'/act'))


class LatentEncoder(nn.Sequential):
    def __init__(self, input_shape, n_latent, num_filters, norm_layer, deterministic=False, use_bias=False,
                 name='Latent Encoder'):
        super(LatentEncoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.deterministic = deterministic
        self.enc = nn.Sequential(input_shape=input_shape, layer_name=name+'/enc')
        self.enc.append(nn.Conv2DLayer(self.enc.output_shape, num_filters, 3, stride=2, no_bias=False, activation='relu',
                                       layer_name=name+'/conv1'))

        self.enc.append(nn.Conv2DLayer(self.enc.output_shape, 2*num_filters, 3, stride=2, no_bias=not use_bias, activation=None,
                                       layer_name=name+'/conv2'))
        self.enc.append(norm_layer(self.enc.output_shape, name+'/norm2'))
        self.enc.append(nn.ActivationLayer(self.enc.output_shape, 'relu', name+'/act2'))

        self.enc.append(nn.Conv2DLayer(self.enc.output_shape, 4*num_filters, 3, stride=2, no_bias=not use_bias, activation=None,
                                       layer_name=name+'/conv3'))
        self.enc.append(norm_layer(self.enc.output_shape, name+'/norm3'))
        self.enc.append(nn.ActivationLayer(self.enc.output_shape, 'relu', name+'/act3'))

        self.enc.append(nn.Conv2DLayer(self.enc.output_shape, 8*num_filters, 3, stride=2, no_bias=not use_bias, activation=None,
                                       layer_name=name+'/conv4'))
        self.enc.append(norm_layer(self.enc.output_shape, name+'/norm4'))
        self.enc.append(nn.ActivationLayer(self.enc.output_shape, 'relu', name+'/act4'))

        self.enc.append(nn.Conv2DLayer(self.enc.output_shape, 8*num_filters, 4, stride=1, no_bias=not use_bias, activation=None,
                                       border_mode='valid', layer_name=name+'/conv5'))
        self.enc.append(norm_layer(self.enc.output_shape, name+'/norm5'))
        self.enc.append(nn.ActivationLayer(self.enc.output_shape, 'relu', name+'/act5'))

        self.enc_mu = nn.Conv2DLayer(self.enc.output_shape, n_latent, 1, no_bias=False, activation=None,
                                     layer_name=name + '/mu')
        self.extend((self.enc, self.enc_mu))

        if not deterministic:
            self.enc_logvar = nn.Conv2DLayer(self.enc.output_shape, n_latent, 1, no_bias=False, activation=None,
                                             layer_name=name + '/logvar')
            self.append(self.enc_logvar)

    def get_output(self, input, *args, **kwargs):
        out = self.enc(input)
        return self.enc_mu(out).flatten(2) if self.deterministic \
            else (self.enc_mu(out).flatten(2), self.enc_logvar(out).flatten(2))


def discriminate(net, crit, fake, real):
    pred_fake = net(fake)
    loss_fake = crit(pred_fake, False)

    pred_true = net(real)
    loss_true = crit(pred_true, True)
    return loss_fake, loss_true, pred_fake, pred_true


class AugmentedCycleGAN:
    def __init__(self, input_shape, n_latent, num_gen_filters=32, num_dis_filters=64, num_enc_filters=32, output_dim=3,
                 use_dropout=False, use_sigmoid=False, use_latent_gan=False, name='Augmented CycleGAN'):
        self.use_latent_gan = use_latent_gan

        self.netG_A_B = CINResnetGen(input_shape, num_gen_filters, n_latent, 3, use_dropout=use_dropout, name=name + 'G_A_B')
        self.netG_B_A = ResnetGen(input_shape, num_gen_filters, 3, use_dropout=use_dropout, name=name + 'G_B_A')

        latent_input_shape = (input_shape[0], input_shape[1] + output_dim) + input_shape[2:]
        self.netE_B = LatentEncoder(latent_input_shape, n_latent, num_enc_filters,
                                    partial(nn.BatchNormLayer, activation=None), deterministic=use_latent_gan,
                                    use_bias=False, name=name + '/E_B')
        self.netD_A = DiscriminatorEdges(self.netG_B_A.output_shape, 32, use_sigmoid=use_sigmoid, name=name+'/D_A')
        self.netD_B = Discriminator(self.netG_A_B.output_shape, num_dis_filters, use_sigmoid=use_sigmoid, name=name+'/D_B')

        if use_latent_gan:
            self.netD_z_B = DiscriminatorLatent((None, n_latent), num_dis_filters, use_sigmoid=use_sigmoid, name=name+'/D_z_B')
        self.gan_loss = partial(nn.gan_loss, div=nn.binary_cross_entropy if use_sigmoid else nn.norm_error)

    def generate_cycle(self, real_A, real_B, z_B):
        fake_B = self.netG_A_B(real_A, z_B)
        fake_A = self.netG_B_A(real_B)
        rec_A = self.netG_B_A(fake_B)

        concat_B_A = T.concatenate((fake_A, real_B), 1)
        stats_z_real_B = self.netE_B(concat_B_A)

        post_z_real_B = stats_z_real_B if self.use_latent_gan \
            else nn.utils.gauss_reparametrize(*stats_z_real_B, clip=-4)

        rec_B = self.netG_A_B(fake_A, post_z_real_B)
        visuals = OrderedDict([('real_B', real_B), ('fake_B', fake_B), ('rec_A', rec_A),
                               ('real_A', real_A), ('fake_A', fake_A), ('rec_B', rec_B)])
        return visuals

    def generate_noisy_cycle(self, real_B, std):
        fake_A = self.netG_B_A(real_B)
        noise_std = std / 127.5
        noise_fake_A = fake_A + nn.utils.srng.normal(fake_A.shape, avg=0., std=noise_std)
        noise_fake_A = T.clip(noise_fake_A, -1, 1)

        concat_B_A = T.concatenate((fake_A, real_B), 1)
        stats_z_real_B = self.netE_B(concat_B_A)

        post_z_real_B = stats_z_real_B if self.use_latent_gan \
            else nn.utils.gauss_reparametrize(*stats_z_real_B, clip=-4)
        rec_B = self.netG_A_B(noise_fake_A, post_z_real_B)
        return rec_B

    def generate_multi(self, real_A, multi_prior_z_B):
        shape = tuple(real_A.shape)
        num = multi_prior_z_B.shape[0] // shape[0]

        multi_real_A = T.tile(real_A.dimshuffle('x', 0, 1, 2, 3), (num, 1, 1, 1, 1))
        multi_real_A = T.reshape(multi_real_A, (-1,) + shape[1:])
        multi_fake_B = self.netG_A_B(multi_real_A, multi_prior_z_B)
        return T.reshape(multi_fake_B, (num, -1) + shape[1:])

    def get_dis_cost(self, real_A, real_B, z_B):
        losses = OrderedDict()

        fake_B = self.netG_A_B(real_A, z_B)
        fake_A = self.netG_B_A(real_B)

        loss_D_fake_A, loss_D_true_A, pred_fake_A, pred_true_A = discriminate(self.netD_A, self.gan_loss, fake_A, real_A)
        loss_D_fake_B, loss_D_true_B, pred_fake_B, pred_true_B = discriminate(self.netD_B, self.gan_loss, fake_B, real_B)

        loss_D_A = .5 * (loss_D_fake_A + loss_D_true_A)
        loss_D_B = .5 * (loss_D_fake_B + loss_D_true_B)

        loss_D = loss_D_A + loss_D_B

        if self.use_latent_gan:
            concat_B_A = T.concatenate((fake_A, real_B), 1)
            mu_z_real_B = self.netE_B(concat_B_A)
            loss_D_fake_z_B, loss_D_true_z_B, pred_fake_z_B, pred_true_z_B = discriminate(self.netD_z_B, self.gan_loss,
                                                                                          mu_z_real_B, z_B)
            loss_D_z_B = .5 * (loss_D_fake_z_B + loss_D_true_z_B)
            loss_D += loss_D_z_B
            losses['loss_D_z_B'] = loss_D_z_B

        losses.update([('D_A', loss_D_A), ('D_B', loss_D_B), ('loss_D', loss_D)])
        preds = OrderedDict([('P_t_A', pred_true_A), ('P_f_A', pred_fake_A), ('P_t_B', pred_true_B),
                             ('P_f_B', pred_fake_B)])
        return losses, preds

    def get_gen_cost(self, real_A, real_B, z_B, lambda_A, lambda_B, lambda_z_B):
        losses = OrderedDict()
        loss_G = 0.

        fake_B = self.netG_A_B(real_A, z_B)
        fake_A = self.netG_B_A(real_B)
        concat_B_A = T.concatenate((fake_A, real_B), 1)
        stats_z_real_B = self.netE_B(concat_B_A)

        if self.use_latent_gan:
            post_z_real_B = stats_z_real_B

            pred_fake_z_B = self.netD_z_B(post_z_real_B)
            loss_G_z_B = self.gan_loss(pred_fake_z_B, True)
            loss_G += loss_G_z_B
            losses['loss_G_z_B'] = loss_G_z_B
        else:
            post_z_real_B = nn.utils.gauss_reparametrize(*stats_z_real_B, clip=-4)

            # measure KLD
            kld_z_B = nn.kld_std_gauss(*stats_z_real_B)
            loss_G += kld_z_B * lambda_z_B
            losses['kld_z_B'] = kld_z_B

        pred_fake_A = self.netD_A(fake_A)
        loss_G_A = self.gan_loss(pred_fake_A, True)

        pred_fake_B = self.netD_B(fake_B)
        loss_G_B = self.gan_loss(pred_fake_B, True)

        # A,z_B -> B -> A,z_B cycle loss
        rec_A = self.netG_B_A(fake_B)
        loss_cycle_A = self.gan_loss(rec_A, real_A)

        # reconstruct z_B from A and fake_B : A ==> z_B <== fake_B
        concat_A_B = T.concatenate((real_A, fake_B), 1)
        stats_z_fake_B = self.netE_B(concat_A_B)

        if self.use_latent_gan:
            loss_cycle_z_B = nn.norm_error(stats_z_fake_B, z_B, 1)
        else:
            # minimize the NLL of original z_B sample
            loss_cycle_z_B = nn.neg_log_prob_gaussian(z_B, *stats_z_real_B)

        # B -> A,z_B -> B cycle loss
        rec_B = self.netG_A_B(fake_A, post_z_real_B)
        loss_cycle_B = nn.norm_error(rec_B, real_B, 1)

        loss_cycle = loss_cycle_A * lambda_A + loss_cycle_B * lambda_B + loss_cycle_z_B * lambda_z_B
        loss_G = loss_G_A + loss_G_B + loss_cycle

        losses.update([('G_A', loss_G_A), ('Cyc_A', loss_cycle_A), ('Cyc_z_B', loss_cycle_z_B), ('G_B', loss_G_B),
                       ('Cyc_B', loss_cycle_B), ('loss_G', loss_G)])
        visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                               ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        return losses, visuals

    def learn(self, real_A, real_B, z_B, lambda_A, lambda_B, lambda_z_B, **kwargs):
        lr = kwargs.pop('lr')
        beta1 = kwargs.pop('beta1')
        max_norm=kwargs.pop('max_norm', 500.)

        dis_losses, dis_preds = self.get_dis_cost(real_A, real_B, z_B)
        gen_losses, _ = self.get_gen_cost(real_A, real_B, z_B, lambda_A, lambda_B, lambda_z_B)

        dis_params = self.netD_A.trainable + self.netD_B.trainable
        if self.use_latent_gan:
            dis_params += self.netD_z_B.trainable
        gen_params = self.netG_A_B.trainable + self.netG_B_A.trainable + self.netE_B.trainable

        updates_dis, _, dis_grads = nn.adam(dis_losses['loss_D'], dis_params, lr, beta1, clip_by_norm=max_norm)
        updates_gen, _, gen_grads = nn.adam(gen_losses['loss_G'], gen_params, lr, beta1, clip_by_norm=max_norm)

        grad_norms = OrderedDict(
            zip([dis_param.name.replace('/', '_') for dis_param in dis_params], [nn.utils.p_norm(dis_grad, 2) for dis_grad in dis_grads]))
        grad_norms.update(
            zip([gen_param.name.replace('/', '_') for gen_param in gen_params], [nn.utils.p_norm(gen_grad, 2) for gen_grad in gen_grads]))
        return updates_dis, updates_gen, dis_losses, dis_preds, gen_losses, grad_norms
