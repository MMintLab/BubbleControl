import os

from bubble_drawing.bubble_learning.models.old.bubble_dynamics_residual_model import BubbleDynamicsResidualModel
from bubble_drawing.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel


class BubblePretrainedCodersDynamicsResidualModel(BubbleDynamicsResidualModel):
    """
    Use as encoder and decoder pretrained bubble_autoencoder modules
    """
    def __init__(self, *args, encoder_version=0, decoder_version=0, **kwargs):
        self.encoder_version = encoder_version
        self.decoder_version = decoder_version
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'bubble_pretrained_coders_dynamics_residual_model'

    def _get_img_encoder(self):
        sizes = self._get_sizes()
        data_path = self.dataset_params['data_name']
        imprint_autoencoder = self._load_model(data_path, load_version=self.encoder_version)
        import pdb; pdb.set_trace()
        img_encoder = None
        # TODO: Set img_embedding size from the loaded model
        return img_encoder

    def _get_img_decoder(self):
        # TODO: Debug
        sizes = self._get_sizes()
        data_path = self.dataset_params['data_name']
        imprint_delta_autoencoder = self._load_model(data_path, load_version=self.decoder_version)
        import pdb; pdb.set_trace()
        img_decoder = None
        # TODO: Set img_embedding size from the loaded model
        return img_decoder

    def _load_model(self, data_path, load_version, load_epoch=None, load_step=None):
            Model = BubbleAutoEncoderModel
            model_name = Model.get_name()
            if load_epoch is None or load_step is None:
                version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                                 'version_{}'.format(load_version), 'checkpoints')
                checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                                  os.path.isfile(os.path.join(version_chkp_path, f))]
                checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
            else:
                # load the specific epoch and step
                checkpoint_path = os.path.join(data_path, 'tb_logs', '{}_logger'.format(model_name),
                                               'version_{}'.format(load_version), 'checkpoints',
                                               'epoch={}-step={}.ckpt'.format(load_epoch, load_step))

            model = Model.load_from_checkpoint(checkpoint_path)
            return model