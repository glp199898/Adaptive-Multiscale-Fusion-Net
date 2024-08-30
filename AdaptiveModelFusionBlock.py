import torch
import torch.nn as nn


class AdaptiveModelFusionBlock(nn.Module):
    def __init__(self):
        super(AdaptiveModelFusionBlock, self).__init__()

        self.w = nn.Parameter(torch.ones(2))

        self.classifier1 = nn.Linear(2048, 512)
        self.classifier2 = nn.Sequential(nn.Linear(512, 32), nn.ReLU(inplace=True))
        self.classifier3 = nn.Linear(32, 512)
        self.classifier4 = nn.Linear(2560, 512)
        self.classifierend = nn.Linear(512, 4)

        self.n = nn.Sequential(nn.Linear(2560, 160), nn.ReLU(inplace=True), nn.Linear(160, 2560))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, y):

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        out3 = x
        out4 = y

        out5 = self.classifier1(out4)

        outadd = out3 + out5
        outadd = self.classifier2(outadd)
        outadd = self.classifier3(outadd)

        weight1 = outadd.softmax(dim=1)

        outcat = torch.cat((out3, out4), dim=1)
        short = outcat
        outcat = self.n(outcat)
        outcat = outcat.softmax(dim=1)
        weight2 = self.classifier4(outcat*short)

        out = w2*weight2 + w1*outadd*weight1

        output = self.classifierend(out)

        return output