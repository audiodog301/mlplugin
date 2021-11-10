#![allow(incomplete_features)]
#![feature(generic_associated_types)]

use baseplug::{Model, Plugin, ProcessContext, UIFloatParam, UIModel, WindowOpenResult};
use neuroflow::activators::Type::Tanh;
use neuroflow::data::DataSet;
use neuroflow::io;
use neuroflow::FeedForward;
use serde::{Deserialize, Serialize};

baseplug::model! {
    #[derive(Debug, Serialize, Deserialize)]
    struct MLModel {
        #[model(min = 0.0, max = 1.0)]
        #[parameter(name = "g")]
        g: f32,
    }
}

impl Default for MLModel {
    fn default() -> Self {
        Self { g: 0.0 }
    }
}

struct ML {
    g: f32,
    network: FeedForward,
}

impl Plugin for ML {
    const NAME: &'static str = "ml";
    const PRODUCT: &'static str = "ml";
    const VENDOR: &'static str = "audiodog301";

    const INPUT_CHANNELS: usize = 2;
    const OUTPUT_CHANNELS: usize = 2;

    type Model = MLModel;

    #[inline]
    fn new(sample_rate: f32, _model: &MLModel) -> Self {
        let mut network = FeedForward::new(&[1, 7, 8, 8, 7, 2]);
        let mut data: DataSet = DataSet::new();
        data.push(&[0.0], &[0.0, 1.0]);
        data.push(&[0.5], &[0.5, 0.0]);
        data.push(&[1.0], &[1.0, 1.0]);

        // Here, we set necessary parameters and train neural network
        // by our DataSet with 50 000 iterations
        network
            .activation(Tanh)
            .learning_rate(0.01)
            .train(&data, 500_000);
        Self {
            g: 0.0,
            network: network,
        }
    }

    #[inline]
    fn process(&mut self, model: &MLModelProcess, ctx: &mut ProcessContext<Self>) {
        let input = &ctx.inputs[0].buffers;
        let output = &mut ctx.outputs[0].buffers;

        for i in 0..ctx.nframes {
            let outs = self.network.calc(&[model.g[i] as f64]);
            let left: f32 = outs[0] as f32;
            let right: f32 = outs[1] as f32;
            output[0][i] = input[0][i] * left;
            output[1][i] = input[1][i] * right;
        }
    }
}

baseplug::vst2!(ML, b"cumy");
