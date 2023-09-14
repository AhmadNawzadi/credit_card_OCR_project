"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from ocr_mods.transformation import TPS_SpatialTransformerNetwork
from ocr_mods.feature_extraction import ResNet_FeatureExtractor
from ocr_mods.sequence_modeling import BidirectionalLSTM
from ocr_mods.prediction import Attention


class Model(nn.Module):

    def __init__(self, Prediction, Transformation, FeatureExtraction, SequenceModeling, imgH, imgW, num_class):
        super(Model, self).__init__()
        self.stages = {'Trans': Transformation, 'Feat': FeatureExtraction,
                       'Seq': SequenceModeling, 'Pred': Prediction}

        """ Transformation """
        if Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=20, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=1)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """

        if FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
                BidirectionalLSTM(256, 256, 256))
            self.SequenceModeling_output = 256
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, 256, num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction
