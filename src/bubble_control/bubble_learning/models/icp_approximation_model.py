import numpy as np
import os
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import pytorch_lightning as pl
import abc
import torchvision
import pytorch3d.transforms as batched_trs
from matplotlib import cm


from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.aux.orientation_trs import QuaternionToAxis
from bubble_control.aux.load_confs import load_object_models
from bubble_control.bubble_learning.aux.pose_loss import PoseLoss


class ICPApproximationModel(pl.LightningModule):

    def __init__(self, input_sizes, num_fcs=2, fc_h_dim=100,
                 skip_layers=None, lr=1e-4, dataset_params=None, activation='relu', load_autoencoder_version=0, object_name='marker', num_to_log=40, autoencoder_augmentation=False):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation
        self.object_name = object_name
        self.autoencoder_augmentation = autoencoder_augmentation
        self.object_model = self._get_object_model()
        self.mse_loss = nn.MSELoss()
        self.pose_loss = PoseLoss(self.object_model)
        self.plane_normal = nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float), requires_grad=False)
        self.num_to_log = num_to_log
        self.autoencoder = self._load_autoencoder(load_version=load_autoencoder_version,
                                                  data_path=self.dataset_params['data_name'])
        self.autoencoder.freeze()
        self.img_embedding_size = self.autoencoder.img_embedding_size  # load it from the autoencoder
        self.pose_estimation_network = self._get_pose_estimation_network()

        self.save_hyperparameters()  # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'icp_approximation_model'

    @property
    def name(self):
        return self.get_name()

    def _get_pose_estimation_network(self):
        input_size = self.img_embedding_size
        output_size = self.input_sizes['object_pose']
        pen_sizes = [input_size] + [self.fc_h_dim]*self.num_fcs + [output_size]
        pen = FCModule(sizes=pen_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return pen

    def _get_object_model(self):
        model_pcs = load_object_models()
        object_model_ar = np.asarray(model_pcs[self.object_name].points)
        return object_model_ar

    def forward(self, imprint):
        img_embedding = self.autoencoder.encode(imprint)
        predicted_pose = self.pose_estimation_network(img_embedding)
        return predicted_pose

    def augmented_forward(self, imprint):
        # Augment the forward by using the reconstructed imprint instead.
        img_embedding = self.autoencoder.encode(imprint)
        imprint_reconstructed = self.autoencoder.decode(img_embedding)
        predicted_pose = self.forward(imprint_reconstructed)
        return predicted_pose

    def _step(self, batch, batch_idx, phase='train'):

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)

        model_output = self.forward(*model_input)
        if self.autoencoder_augmentation:
            # TODO: This maybe has a limit on memory allocation (we x2 the required memory). Consider adding losses up instead.
            augmented_model_output = self.augmented_forward(*model_input)
            model_output = torch.cat([model_output, augmented_model_output], dim=0)
            ground_truth = tuple([torch.cat([ground_truth_i, ground_truth_i], dim=0) for ground_truth_i in ground_truth])

        loss = self._compute_loss(model_output, *ground_truth)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        self._log_object_pose_images(obj_pose_pred=model_output[:self.num_to_log], obj_pose_gth=ground_truth[0][:self.num_to_log], phase=phase)
        self._log_imprint(batch, batch_idx=batch_idx, phase=phase)
        return loss

    def _get_sizes(self):
        sizes = {}
        sizes.update(self.input_sizes)
        sizes['dyn_input_size'] = self._get_dyn_input_size(sizes)
        sizes['dyn_output_size'] = self._get_dyn_output_size(sizes)
        return sizes

    def get_input_keys(self):
        input_keys = ['imprint']
        return input_keys

    @abc.abstractmethod
    def get_model_output_keys(self):
        output_keys = ['object_pose']
        return output_keys

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_model_input(self, sample):
        input_key = self.get_input_keys()
        model_input = [sample[key] for key in input_key]
        model_input = tuple(model_input)
        return model_input

    def get_model_output(self, sample):
        output_keys = self.get_model_output_keys()
        model_output = [sample[key] for key in output_keys]
        model_output = tuple(model_output)
        return model_output

    def _compute_loss(self, obj_pose_pred, obj_pose_gth):
        # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
        axis_angle_pred = obj_pose_pred[..., 3:]
        R_pred = batched_trs.axis_angle_to_matrix(axis_angle_pred)
        t_pred = obj_pose_pred[..., :3]
        axis_angle_gth = obj_pose_gth[..., 3:]
        R_gth = batched_trs.axis_angle_to_matrix(axis_angle_gth)
        t_gth = obj_pose_gth[..., :3]
        pose_loss = self.pose_loss(R_1=R_pred, t_1=t_pred, R_2=R_gth, t_2=t_gth)
        # pose_loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        loss = pose_loss
        return loss


    # AUX FUCTIONS -----------------------------------------------------------------------------------------------------

    def _load_autoencoder(self, load_version, data_path, load_epoch=None, load_step=None):
        Model = BubbleAutoEncoderModel
        model_name = Model.get_name()
        if load_epoch is None or load_step is None:
            version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                             'version_{}'.format(load_version), 'checkpoints')
            checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                              os.path.isfile(os.path.join(version_chkp_path, f))]
            checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
        else:
            checkpoint_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                           'version_{}'.format(load_version), 'checkpoints',
                                           'epoch={}-step={}.ckpt'.format(load_epoch, load_step))

        model = Model.load_from_checkpoint(checkpoint_path)

        return model

    def _log_imprint(self, batch, batch_idx, phase):
        if self.current_epoch == 0 and batch_idx == 0:
            imprint_t = batch['imprint'][:self.num_imprints_to_log]
            self.logger.experiment.add_image('imprint_{}'.format(phase), self._get_image_grid(imprint_t),
                                             self.global_step)
            if self.autoencoder_augmentation:
                reconstructed_imprint_t = self.autoencoder.decode(self.autoencoder.encode(imprint_t))
                self.logger.experiment.add_image('imprint_reconstructed_{}'.format(phase),
                                                 self._get_image_grid(reconstructed_imprint_t), self.global_step)


    def _log_object_pose_images(self, obj_pose_pred, obj_pose_gth, phase):
        obj_trans_pred = obj_pose_pred[..., :3]
        obj_rot_pred = obj_pose_pred[..., 3:]
        obj_rot_angle_pred = self._get_angle_from_axis_angle(obj_rot_pred, self.plane_normal)
        obj_trans_gth = obj_pose_gth[..., :3]
        obj_rot_gth = obj_pose_gth[..., 3:]
        obj_rot_angle_gth = self._get_angle_from_axis_angle(obj_rot_gth, self.plane_normal)
        images = self._get_pose_images(obj_trans_pred, obj_rot_angle_pred, obj_trans_gth, obj_rot_angle_gth)
        grid = torchvision.utils.make_grid(images)
        self.logger.experiment.add_image('pose_estimation_{}'.format(phase), grid, self.global_step)

    def _get_angle_from_axis_angle(self, orientation, plane_normal):
        if orientation.shape[-1] == 4:
            q_to_ax = QuaternionToAxis()
            axis_angle = torch.from_numpy(q_to_ax._tr(orientation.detach().numpy()))
        else:
            axis_angle = orientation
        normal_axis_angle = torch.einsum('bi,i->b', axis_angle, plane_normal).unsqueeze(-1) * plane_normal.unsqueeze(0)
        angle = torch.norm(normal_axis_angle, dim=-1)
        if torch.sum(torch.isnan(angle)) != 0:
            import pdb; pdb.set_trace()
        return angle

    def _get_pose_images(self, trans_pred, rot_angle_pred, trans_gth, rot_angle_gth):
        images = []
        for i in range(len(trans_pred)):
            img = np.zeros([100, 100, 3], dtype=np.uint8)
            img.fill(100)
            pred_param = self._find_rect_param(trans_pred[i], rot_angle_pred[i], img)
            color_p = (255, 0, 0)
            self._draw_angled_rec(*pred_param, color_p, img)
            gth_param = self._find_rect_param(trans_gth[i], rot_angle_gth[i], img)
            color_gth = (0, 0, 255)
            self._draw_angled_rec(*gth_param, color_gth, img)
            img = torch.tensor(img)
            img = img.permute(2, 0, 1)
            images.append(img)
        return images

    def _find_rect_param(self, trans, rot, img):
        height = 0.06 * 100 / 0.15
        width = 0.015 * 100 / 0.15
        center_x = img.shape[0] / 2 + trans[0] * 10 / 0.15
        center_y = img.shape[1] / 2 + trans[1] * 10 / 0.15
        return center_x, center_y, width, height, rot.item()

    def _draw_angled_rec(self, x0, y0, width, height, angle, color, img):
        b = np.cos(angle) * 0.5
        a = np.sin(angle) * 0.5
        pt0 = (int(x0 - a * height - b * width),
            int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width),
            int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        cv2.line(img, pt0, pt1, color, 3)
        cv2.line(img, pt1, pt2, color, 3)
        cv2.line(img, pt2, pt3, color, 3)
        cv2.line(img, pt3, pt0, color, 3)

    def _get_image_grid(self, batched_img, cmap='jet'):
        # reshape the batched_img to have the same imprints one above the other
        batched_img = batched_img.detach().cpu()
        batched_img_r = batched_img.reshape(*batched_img.shape[:1], -1, *batched_img.shape[3:]) # (batch_size, 2*W, H)
        # Add padding
        padding_pixels = 5
        batched_img_padded = F.pad(input=batched_img_r,
                                   pad=(padding_pixels, padding_pixels, padding_pixels, padding_pixels),
                                   mode='constant',
                                   value=0)
        batched_img_cmap = self._cmap_tensor(batched_img_padded, cmap=cmap) # size (..., w,h, 3)
        num_dims = len(batched_img_cmap.shape)
        grid_input = batched_img_cmap.permute(*np.arange(num_dims-3), -1, -3, -2)
        grid_img = torchvision.utils.make_grid(grid_input)
        return grid_img

    def _cmap_tensor(self, img_tensor, cmap='jet'):
        cmap = cm.get_cmap(cmap)
        mapped_img_ar = cmap(img_tensor/torch.max(img_tensor)) # (..,w,h,4)
        mapped_img_ar = mapped_img_ar[..., :3] # (..,w,h,3) -- get rid of the alpha value
        mapped_img = torch.tensor(mapped_img_ar).to(self.device)
        return mapped_img


class FakeICPApproximationModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imprint):
        fake_pose_shape = imprint.shape[:-3] + (6,)
        fake_pose = torch.zeros(fake_pose_shape, device=imprint.device, dtype=imprint.dtype) # encoded as axis-angle
        return fake_pose

    @classmethod
    def get_name(cls):
        return 'fake_icp_approximation_model'

    @property
    def name(self):
        return self.get_name()

