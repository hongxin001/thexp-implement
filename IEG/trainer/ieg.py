import torch
import numpy as np
from thexp import Trainer, callbacks, Params
from torch.nn import functional as F
from torch.optim import SGD
from trick import onehot

from . import GlobalParams


class IEG(Trainer, callbacks.TrainCallback):

    def __init__(self, params: GlobalParams = None):
        super().__init__(params)
        self.params = params

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)

    def datasets(self, params: GlobalParams):
        super().datasets(params)

    def models(self, params: GlobalParams):
        from ..wideresnet import wideresnet
        self.net = wideresnet(28, 10, params.num_classes)
        self.optim = SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def train_batch(self, eidx, idx, global_step, batch_data, params: GlobalParams, device: torch.device):
        pass

    def unsupervised_loss(self, images, aug_images, logits, probe_images, probe_labels):
        """Creates unsupervised losses.

        Here we create two cross-entropy losses and a KL-loss defined in the paper.

        Returns:
          A list of losses.
        """

        # if FLAGS.ce_factor == 0 and FLAGS.consistency_factor == 0:
        #     return [tf.constant(0, tf.float32), tf.constant(0, tf.float32)]

        im_shape = (-1, int(probe_images.shape[1]), int(probe_images.shape[2]),
                    int(probe_images.shape[3]))

        aug_logits = self.net(aug_images, name='model', training=True)

        n_probe_to_mix = aug_images.shape[0]

        # probe = tf.tile(tf.constant([[10.]]), [1, tf.shape(probe_images)[0]])
        # idx = tf.squeeze(tf.random.categorical(probe, n_probe_to_mix)) # 因为是按恒定概率分布采样，因此取相同的
        idx = torch.randint(0, probe_images.shape[0], n_probe_to_mix)

        # l_images = torch.reshape(tf.gather(probe_images, idx), im_shape)
        # l_labels = torch.reshape(tf.gather(probe_labels, idx), (-1,))
        l_images = torch.reshape(torch.gather(probe_images, idx), im_shape)
        l_labels = torch.reshape(torch.gather(probe_labels, idx), (-1,))

        u_logits = torch.cat([logits, aug_logits], dim=0)
        u_images = torch.cat([images, aug_images], dim=0)

        losses = []
        if self.params.ce_factor > 0:
            ce_min_loss = self.crossentropy_minimize(u_logits, u_images, l_images,
                                                     l_labels)
            losses.append(ce_min_loss)
        else:
            losses.append(0.0)

        if self.params.consistency_factor > 0:
            consis_loss = self.consistency_loss(logits, aug_logits)
            losses.append(consis_loss)
        else:
            losses.append(0.0)

        return losses

    def consistency_loss(self, logit, aug_logit):

        def kl_divergence(q_logits, p_logits):
            q = F.softmax(q_logits, dim=1)
            per_example_kl_loss = q * (
                    F.log_softmax(q_logits, dim=1) - F.log_softmax(p_logits, dim=1))
            return per_example_kl_loss.mean() * self.params.num_classes

        return kl_divergence(logit.detach(), aug_logit) * self.params.consistency_factor

    def guess_label(self, logit, temp=0.5):
        logit = tf.reshape(logit, [-1, self.dataset.num_classes])
        logit = tf.split(logit, self.nu, axis=0)
        logit = [logit_norm(x) for x in logit]
        logit = tf.concat(logit, 0)
        ## Done with logit norm
        p_model_y = tf.reshape(
            tf.nn.softmax(logit), [self.nu, -1, self.dataset.num_classes])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)

        p_target = tf.pow(p_model_y, 1.0 / temp)
        p_target /= tf.reduce_sum(p_target, axis=1, keepdims=True)

        return p_target

    def augment(self, sup_imgs, un_sup_imgs, sup_targets, targets_u, beta):
        targets = torch.cat([sup_targets, targets_u])
        imgs = torch.cat((sup_imgs, un_sup_imgs))
        idx = torch.randperm(imgs.size(0))
        input_a, input_b = imgs, imgs[idx]
        target_a, target_b = targets, targets[idx]

        l = np.random.beta(beta, beta)
        l = max(l, 1 - l)
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        return mixed_input, mixed_target

    def crossentropy_minimize(self,
                              u_logits,
                              u_images,
                              l_images,
                              l_labels,
                              u_labels=None):
        """Cross-entropy optimization step implementation for TPU."""
        batch_size = self.params.batch_size
        guessed_label = self.guess_label(u_logits)
        self.guessed_label = guessed_label

        guessed_label = torch.reshape(guessed_label.detach(), shape=(-1, self.params.num_classes))

        l_labels = torch.reshape(
            onehot(l_labels, self.params.num_classes),
            shape=(-1, self.params.num_classes))

        augment_images, augment_labels = self.augment(
            l_images, u_images,
            l_labels, guessed_label * self.params.nu,
            self.params.beta)

        logit = self.net(augment_images)

        zbs = batch_size * 2
        halfzbs = batch_size

        split_pos = [l_images.shape[0], halfzbs, halfzbs]

        logit = [logit_norm(lgt) for lgt in torch.split_with_sizes(logit, split_pos)]
        u_logit = torch.cat(logit[1:], dim=0)

        split_pos = [l_images.shape[0], zbs]
        l_augment_labels, u_augment_labels = torch.split_with_sizes(
            augment_labels, split_pos)

        u_loss = tf.losses.softmax_cross_entropy(u_augment_labels, u_logit)
        l_loss = tf.losses.softmax_cross_entropy(l_augment_labels, logit[0])

        loss = tf.math.add(
            l_loss, u_loss * FLAGS.ce_factor, name='crossentropy_minimization_loss')

        return loss


def logit_norm(v, use_logit_norm=True):
    if not use_logit_norm:
        return v
    return v * tf.math.rsqrt(tf.reduce_mean(tf.square(v)) + 1e-8)