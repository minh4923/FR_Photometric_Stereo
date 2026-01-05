import torch
import torch.nn as nn
from .focalloss.FocalLoss import FocalLoss
from .WeightClassMagLoss import WeightClassMagLoss
from going_modular.model.head.id import MagLinear 

class ConcatMultiTaskLoss(torch.nn.Module):
    def __init__(self, metadata_path:str, conf:dict):
        super(ConcatMultiTaskLoss, self).__init__()
        # ID Loss
        self.id_head = MagLinear(512, conf['num_classes'])
        self.id_loss = WeightClassMagLoss(metadata_path)
        
        # Aux Loss
        self.gender_loss = FocalLoss(alpha_weights={0:0.916, 1:0.084}, gamma_weights={0:2, 1:0}, num_classes=2)
        self.spectacles_loss = FocalLoss(alpha_weights={0: 0.28, 1: 0.72}, gamma_weights={0:0, 1:1}, num_classes=2)
        self.facial_hair_loss = FocalLoss(alpha_weights={0:0.3, 1:0.7}, gamma_weights={0:0, 1:1}, num_classes=2)
        self.pose_loss = FocalLoss(alpha_weights={0: 0.0263, 1: 0.9737}, gamma_weights={0:0, 1:2.5}, num_classes=2)
        self.emotion_loss = FocalLoss(alpha_weights={0:0.232, 1:0.768}, gamma_weights={0:0, 1:1}, num_classes=2)
        
        self.conf = conf

    def forward(self, x_fused, aux_outputs, y):
        # Tách nhãn
        label_id, label_gender, label_spectacles = y[:, 0], y[:, 1], y[:, 2]
        label_facial_hair, label_pose, label_emotion = y[:, 3], y[:, 4], y[:, 5]
        
        # ID Loss
        x_id_logits, x_id_norm = self.id_head(x_fused)
        loss_id = self.id_loss(x_id_logits, label_id, x_id_norm)
        
        # Aux Loss
        loss_spec = self.spectacles_loss(aux_outputs[0][0], label_spectacles)
        loss_hair = self.facial_hair_loss(aux_outputs[1][0], label_facial_hair)
        loss_emo = self.emotion_loss(aux_outputs[2][0], label_emotion)
        loss_pose = self.pose_loss(aux_outputs[3][0], label_pose)
        loss_gender = self.gender_loss(aux_outputs[4][0], label_gender)
        
        total_loss = loss_id + \
                     loss_gender * self.conf['loss_gender_weight'] + \
                     loss_emo * self.conf['loss_emotion_weight'] + \
                     loss_pose * self.conf['loss_pose_weight'] + \
                     loss_hair * self.conf['loss_facial_hair_weight'] + \
                     loss_spec * self.conf['loss_spectacles_weight']
                     
        return total_loss, {
            'id': loss_id, 'gender': loss_gender, 'emotion': loss_emo,
            'pose': loss_pose, 'facial_hair': loss_hair, 'spectacles': loss_spec
        }