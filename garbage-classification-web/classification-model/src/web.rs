use model::{MinResNet, MinResNetRecord};

use alloc::vec::Vec;

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{BinBytesRecorder, HalfPrecisionSettings, Recorder};

use burn::tensor::Tensor;
use wasm_bindgen::prelude::*;

pub type Backend = NdArray<f32>;

const CHANNEL: usize = 3;
const WIDTH: usize = 256;
const HEIGHT: usize = 256;

const STATE_ENCODED: &[u8] = include_bytes!("../ckpt/2040-77.bin");

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct GarbageClassification {
    model: MinResNet<Backend>,
}

#[wasm_bindgen]
impl GarbageClassification {
    /// Constructor called by JavaScripts with the new keyword.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let model: MinResNet<Backend> = MinResNet::init(&Default::default());
        let record: MinResNetRecord<Backend> = BinBytesRecorder::<HalfPrecisionSettings>::default()
            .load(STATE_ENCODED.into(), &Default::default())
            .expect("Failed to decode state");
        Self {
            model: model.load_record(record),
        }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 256x256 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
    /// * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
    ///
    pub async fn inference(&self, input: &[f32]) -> Vec<f32> {
        let device = Default::default();
        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input = Tensor::<Backend, 1>::from_floats(input, &device).reshape([
            1,
            HEIGHT,
            WIDTH,
            CHANNEL + 1,
        ]);
        // Drop alpha channel
        let input = input.slice([None, None, None, Some((0, -1))]);
        // Permute the tensor to [batch, channel, width, height]
        let input = input.permute([0, 3, 2, 1]);

        // Normalize input: make between [0,1] and make the mean=0.5 and std=0.5
        let input = input / 255;
        let input = (input - 0.5) / 0.5;

        // Run the tensor input through the model
        let output: Tensor<Backend, 2> = self.model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let output = burn::tensor::activation::softmax(output, 1);

        // Flatten output tensor with [1, 6] shape into boxed slice of [f32]
        let output = output.into_data_async().await;

        output.to_vec().unwrap()
    }
}
