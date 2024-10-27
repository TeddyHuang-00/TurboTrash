use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    resample: Conv2d<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn init(device: &B::Device, in_channels: usize, out_channels: usize) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3; 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let conv2 = Conv2dConfig::new([out_channels; 2], [3; 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);
        let resample = Conv2dConfig::new([in_channels, out_channels], [1; 2])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        Self {
            conv1,
            conv2,
            bn1,
            bn2,
            resample,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let _x = self.resample.forward(x.clone());
        let y = self.conv1.forward(x);
        let y = self.bn1.forward(y);
        let y = Relu.forward(y);
        let y = self.conv2.forward(y);
        let y = self.bn2.forward(y);
        let y = y + _x;
        let y = Relu.forward(y);
        y
    }
}

#[derive(Module, Debug)]
pub struct MinResNet<B: Backend> {
    pool: MaxPool2d,
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    layer1: ResidualBlock<B>,
    layer2: ResidualBlock<B>,
    layer3: ResidualBlock<B>,
    layer4: ResidualBlock<B>,
    layer5: ResidualBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> MinResNet<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let dim_hiddens = [8, 16, 32, 64, 128, 256];
        let pool = MaxPool2dConfig::new([3; 2])
            .with_strides([2; 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
        let conv1 = Conv2dConfig::new([3, dim_hiddens[0]], [7; 2])
            .with_bias(false)
            .with_stride([2; 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .init(device);
        let bn1 = BatchNormConfig::new(dim_hiddens[0]).init(device);
        let layer1 = ResidualBlock::init(device, dim_hiddens[0], dim_hiddens[1]);
        let layer2 = ResidualBlock::init(device, dim_hiddens[1], dim_hiddens[2]);
        let layer3 = ResidualBlock::init(device, dim_hiddens[2], dim_hiddens[3]);
        let layer4 = ResidualBlock::init(device, dim_hiddens[3], dim_hiddens[4]);
        let layer5 = ResidualBlock::init(device, dim_hiddens[4], dim_hiddens[5]);
        let avgpool = AdaptiveAvgPool2dConfig::new([1; 2]).init();
        let fc = LinearConfig::new(dim_hiddens[4], 6).init(device);
        Self {
            pool,
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            avgpool,
            fc,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = Relu.forward(x);
        let x = self.pool.forward(x);
        let x = self.layer1.forward(x);
        let x = self.pool.forward(x);
        let x = self.layer2.forward(x);
        let x = self.pool.forward(x);
        let x = self.layer3.forward(x);
        let x = self.pool.forward(x);
        let x = self.layer4.forward(x);
        let x = self.pool.forward(x);
        let x = self.layer5.forward(x);
        let x = self.avgpool.forward(x);
        let x = x.flatten(1, 3);
        let x = self.fc.forward(x);
        x
    }
}
