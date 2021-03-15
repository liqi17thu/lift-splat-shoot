from segmentTool.models import *
builder = ModelBuilder()


class TransformModule(nn.Module):
    def __init__(self, dim=25, num_view=8):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.dim = dim
        self.mat_list = nn.ModuleList()
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU(),
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.dim * self.dim,])
        view_comb = self.mat_list[0](x[:, 0])
        for index in range(x.size(1))[1:]:
            view_comb += self.mat_list[index](x[:, index])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb


class SumModule(nn.Module):
    def __init__(self):
        super(SumModule, self).__init__()

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = torch.sum(x, dim=1, keepdim=False)
        return x


class VPNModel(nn.Module):
    def __init__(self):
        super(VPNModel, self).__init__()
        self.num_views = 6
        self.transform_type = 'fc'
        print('Views number: ' + str(self.num_views))
        print('Transform Type: ', self.transform_type)
        self.encoder = builder.build_encoder(
                    arch='resnet18',
                    fc_dim=256,
                )
        if self.transform_type == 'fc':
            self.transform_module = TransformModule(dim=self.output_size, num_view=self.num_views)
        elif self.transform_type == 'sum':
            self.transform_module = SumModule()
        self.decoder = builder.build_decoder(
                    arch='ppm_bilinear',
                    fc_dim=256,
                    num_class=4,
                    use_softmax=False,
                )

    def forward(self, x, return_feat=False):
        B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                            + list(x.size()[2:])).size()

        x = x.view(B*N, C, H, W)
        x = self.encoder(x)[0]
        x = x.view([B, N] + list(x.size()[1:]))
        x = self.transform_module(x)
        if return_feat:
            x, feat = self.decoder([x], return_feat=return_feat)
        else:
            x = self.decoder([x])
        x = x.transpose(1, 2).transpose(2, 3).contiguous()
        if return_feat:
            feat = feat.transpose(1, 2).transpose(2, 3).contiguous()
            return x, feat
        return x


if __name__ == '__main__':
    ori_vpn = VPNModel()
    x = torch.randn(4, 6, 3, 128, 352)
    out = ori_vpn(x)
    print(out.shape)