"""
SegNet implementation is taken from github: https://github.com/say4n/pytorch-segnet/blob/f7738c6bce384b54fcbb3fe8aff02736d6ec2285/src/model.py#L333
and modified to develope the proposed model SegNet+GRU.
"""

from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import pprint
F = nn.functional
DEBUG = False


vgg16_dims = [
                    (64, 64, 'M'),                                # Stage - 1
                    (128, 128, 'M'),                              # Stage - 2
                    (256, 256, 256,'M'),                          # Stage - 3
                    (512, 512, 512, 'M'),                         # Stage - 4
                    (512, 512, 512, 'M')                          # Stage - 5
            ]

decoder_dims = [
                    ('U', 512, 512, 512),                         # Stage - 5
                    ('U', 512, 512, 512),                         # Stage - 4
                    ('U', 256, 256, 256),                         # Stage - 3
                    ('U', 128, 128),                              # Stage - 2
                    ('U', 64, 64)                                 # Stage - 1
                ]

				
class SegNetGRU_Symmetric_columns_UltimateShare(nn.Module):
    '''
    The gru units are shared between encoding and decoding
    '''

    def __init__(self, input_channels=3, num_classes=2, VGG_pretrained=True):
        super(SegNetGRU_Symmetric_columns_UltimateShare, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.num_channels = input_channels

        self.vgg16 = models.vgg16(pretrained=VGG_pretrained)

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])

        # finish stage1
        # Stage 2
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        # concat gru result before the last conv within the stage will adjust the depth size to be compatible with the corresponding max up-pool
        # GRU stage 2
        self.gru_2_row = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128 + 128,  # add GRU feature map
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        # *********** MAX POOL ***********************************#
        # finish stage2 rnn_1 output will be concat with the following stage,hence, depth is doubled
        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,  # *2 due to GRU result
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        # GRU stage 3
        self.gru_3_row = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256 + 256,  # add GRU feature map
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        # ****************** MAX POOL ****************************#
        # self.rnn_2_row = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.rnn_2_column = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # finish stage3
        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        # GRU stage 4
        self.gru_4_row = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)

        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512 + 512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        # ****************** MAX POOL ****************************#
        # finish stage4
        self.encoder_conv_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        # GRU stage 5
        self.gru_5_row = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)

        self.encoder_conv_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512 + 512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        # ****************** MAX POOL ****************************#
        # self.rnn_4_row = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        # self.rnn_4_column = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        # finish last stage5
        if VGG_pretrained:
            self.init_vgg_weigts()

        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        # de GRU stage 5
        # GRU from the encoder will be applied
        self.decoder_convtr_41 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512+512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_40 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_32 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        # de GRU stage 4
        # GRU units from decoder will be applied
        self.decoder_convtr_31 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512+512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_30 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_22 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        # de GRU stage 3
        # GRU units from decoder will be applied
        self.decoder_convtr_21 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256+256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_20 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        # de GRU stage 2
        # GRU units from decoder will be applied
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128 + 128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=self.num_classes,
                               kernel_size=3,
                               padding=1)
        ])

    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """

        # Encoder

        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        # GRU at stage 2
        x_10 = self.gruUnit(x_10, self.gru_2_row,self.gru_2_row)
        # Finish GRU at stage 2
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        # GRU at stage 3
        x_21 = self.gruUnit(x_21, self.gru_3_row,self.gru_3_row)
        # Finish GRU at stage 3
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        # GRU at stage 4
        x_31 = self.gruUnit(x_31, self.gru_4_row,self.gru_4_row)
        # Finish GRU at stage 4
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        # GRU at stage 5
        x_41 = self.gruUnit(x_41, self.gru_5_row,self.gru_5_row)
        # Finish GRU at stage 5
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)

        # Decoder

        dim_d = x_4.size()

        # Decoder Stage - 5
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        # de GRU at stage 5
        x_42d = self.gruUnit(x_42d, self.gru_5_row,self.gru_5_row)
        # Finish de GRU at stage 5
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))
        dim_4d = x_40d.size()

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        # de GRU at stage 4
        x_32d = self.gruUnit(x_32d, self.gru_4_row,self.gru_4_row)
        # Finish de GRU at stage 4
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        # de GRU at stage 3
        x_22d = self.gruUnit(x_22d, self.gru_3_row, self.gru_3_row)
        # Finish de GRU at stage 3
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        # de GRU at stage 2
        x_11d = self.gruUnit(x_11d, self.gru_2_row,self.gru_2_row)
        # Finish de GRU at stage 2
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()

        # x_softmax = F.softmax(x_00d, dim=1)

        if DEBUG:
            print("dim_0: {}".format(dim_0))
            print("dim_1: {}".format(dim_1))
            print("dim_2: {}".format(dim_2))
            print("dim_3: {}".format(dim_3))
            print("dim_4: {}".format(dim_4))

            print("dim_d: {}".format(dim_d))
            print("dim_4d: {}".format(dim_4d))
            print("dim_3d: {}".format(dim_3d))
            print("dim_2d: {}".format(dim_2d))
            print("dim_1d: {}".format(dim_1d))
            print("dim_0d: {}".format(dim_0d))

        return x_00d#, x_softmax

    def apply_GRU(self,x,gru_model):
        featureMap_shape = x.shape  # store the original feature shape
        x_gru = x.reshape(featureMap_shape[0], -1, featureMap_shape[1])  # (batch_size, timesteps, channels"feature vectore for each Xt")
        x_gru, h_11 = gru_model(x_gru)  # get results after feeding it viewed feature map
        x_gru = x_gru.reshape(featureMap_shape[0], -1, featureMap_shape[2], featureMap_shape[3])  # convert GRU result shape
        return x_gru

    def gruUnit(self, x, gru_model_row,gru_model_column):

        #rows GRU
        x_gru_row = self.apply_GRU(x,gru_model_row)
        #columns GRU
        x_transposed = x.transpose(2,3)
        x_gru_column = self.apply_GRU(x_transposed, gru_model_column)
        x_gru_column = x_gru_column.transpose(2,3)

        x_cated = torch.cat((x, x_gru_row,x_gru_column), 1)
        return x_cated

    def init_vgg_weigts(self):
        assert self.encoder_conv_00[0].weight.size() == self.vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = self.vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == self.vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = self.vgg16.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == self.vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = self.vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == self.vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = self.vgg16.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == self.vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = self.vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == self.vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = self.vgg16.features[5].bias.data
        # wieght size of this conv is different from VGG16 due to our modification of in_channel and out_channel
        # assert self.encoder_conv_11[0].weight.size() == self.vgg16.features[7].weight.size()
        # self.encoder_conv_11[0].weight.data = self.vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == self.vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = self.vgg16.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == self.vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = self.vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == self.vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = self.vgg16.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == self.vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = self.vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == self.vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = self.vgg16.features[12].bias.data

        # assert self.encoder_conv_22[0].weight.size() == self.vgg16.features[14].weight.size()
        # self.encoder_conv_22[0].weight.data = self.vgg16.features[14].weight.data
        # assert self.encoder_conv_22[0].bias.size() == self.vgg16.features[14].bias.size()
        # self.encoder_conv_22[0].bias.data = self.vgg16.features[14].bias.data

        assert self.encoder_conv_30[0].weight.size() == self.vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = self.vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == self.vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = self.vgg16.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == self.vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = self.vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == self.vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = self.vgg16.features[19].bias.data

        # assert self.encoder_conv_32[0].weight.size() == self.vgg16.features[21].weight.size()
        # self.encoder_conv_32[0].weight.data = self.vgg16.features[21].weight.data
        # assert self.encoder_conv_32[0].bias.size() == self.vgg16.features[21].bias.size()
        # self.encoder_conv_32[0].bias.data = self.vgg16.features[21].bias.data

        assert self.encoder_conv_40[0].weight.size() == self.vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = self.vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == self.vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = self.vgg16.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == self.vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = self.vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == self.vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = self.vgg16.features[26].bias.data

        # assert self.encoder_conv_42[0].weight.size() == self.vgg16.features[28].weight.size()
        # self.encoder_conv_42[0].weight.data = self.vgg16.features[28].weight.data
        # assert self.encoder_conv_42[0].bias.size() == self.vgg16.features[28].bias.size()
        # self.encoder_conv_42[0].bias.data = self.vgg16.features[28].bias.data


if __name__=='__main__':
    model = SegNetGRU_Symmetric_columns_UltimateShare(VGG_pretrained=True)
    input = torch.rand((4,3,100,100))
    output = model(input)
    print('output dim = ',output.shape)
    # of param of UltimateShare= 180007914
    number_of_parameters= total_params = sum(param.numel() for param in model.parameters())
    print('# of param of UltimateShare=',number_of_parameters)

