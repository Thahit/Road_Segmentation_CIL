import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvX(nn.Module):
    """(convolution)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class TrippleConvX(nn.Module):
    """(convolution)"""

    def __init__(self, dim):
        super().__init__()

        self.tripple_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, bias=False, 
                      groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim , dim*4, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.tripple_conv(x)


class DownX(nn.Module):
    """Downscaling + conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0, bias=False,# 
                      stride= 2)# downsample payer
        self.tripple = TrippleConvX(out_channels)

    def forward(self, x):
        x = self.down_conv(x)
        return self.tripple(x)


class UpX(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, extra_in = False):
        # extra_in is an extra bool indicating that the pre_conv should have x2 its normal size
        # (this is due to the number of channels not being x2 every time you go down)
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.pre_conv = nn.Conv2d(in_channels*2 if extra_in else in_channels, out_channels, kernel_size=1, padding=0, bias=False)# work with the re-added part from the down layers
        self.conv = TrippleConvX(out_channels)

    def forward(self, x1, x2):# from last layer, from down layer
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.pre_conv(x)
        return self.conv(x) + x


class OutConvX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvX, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Att_block(nn.Module):#https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Att_block,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out#,attention

class UNext(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNext, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConvX(n_channels, 64))
        # less channels? 
        self.down1 = (DownX(64, 128))
        self.down2 = (DownX(128, 256))
        self.down3 = (DownX(256, 512))
        self.down4 = (DownX(512, 512))
        self.att = Att_block(512, 512)
        self.up1 = (UpX(512, 512, extra_in=True))
        self.up2 = (UpX(512, 256))
        self.up3 = (UpX(256, 128))
        self.up4 = (UpX(128, 64))
        self.outc = (OutConvX(64, n_classes))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.att(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x) 
        return x

    def use_checkpointing(self):#unused
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def get_model():
    model = UNext(3, 1)
    return model