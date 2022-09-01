import torch
from torch import nn


class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        # (kalpitthakkar's comment)
        # We take the mean of all vectors in the input to this layer
        # along dimensions 2 and 3.
        # 1. x_t is the output obtained by performing spatial pooling,
        #    i.e. taking the mean of all the joints in a frame. Can
        #    think of this as finding the centroid of each frame's skeleton.
        # 2. x_v is the output obtained by performing temporal pooling,
        #    i.e. taking the mean of each joint's location across time.
        #    You can think of this as the mean skeleton of the entire video.
        # The final shape of x_t and x_v is (N, C, K, 1), where K = T or V.
        # We transpose x_v across dimensions 2 and 3 as we want V in dimension
        # index 2 rather than 3.
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        # (kalpitthakkar's comment)
        # Referring to the Figure 7 given in the paper, these two matrices
        # are concatenated across the dimension have K in the final shape
        # (index 2). This concatenated output is fed to a FC layer for
        # combining all the signals from spatial and temporal dimensions.
        # Continuing further in the figure, we divide the attented output
        # from FC layer into x_t and x_v again. We then apply separate 1x1
        # 2D convolutions to generate new features and apply sigmoid act
        # function to get probabilities for each output.
        # Notice carefully, we transposed x_v again, which means when we
        # multiply the two attended outputs from 2D convs, our final
        # output shape will be (N, C, T, V).
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


class Part_Att(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel*len(self.parts), kernel_size=1, bias=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)


class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)


class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)


class Joint_Att(nn.Module):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fcn(x.transpose(1, 3)).transpose(1, 3)
