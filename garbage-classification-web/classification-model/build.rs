/// This build script does the following:
/// 1. Loads PyTorch weights into a model record.
/// 2. Saves the model record to a file using the `NamedMpkFileRecorder`.
use std::path::Path;

use burn::{
    backend::NdArray,
    record::{BinBytesRecorder, HalfPrecisionSettings, Recorder},
};
use burn_import::pytorch::PyTorchFileRecorder;
use model;

// Basic backend type (not used directly here).
type B = NdArray<f32>;

fn main() {
    println!("cargo::rerun-if-changed=pytorch-ckpt");
    let device = Default::default();

    let checkpoint_dir = Path::new("pytorch-ckpt");
    for entry in std::fs::read_dir(checkpoint_dir).expect("Failed to read checkpoint directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_str().unwrap();
            println!("Loading checkpoint: {}", name);
            let record: model::MinResNetRecord<B> =
                PyTorchFileRecorder::<HalfPrecisionSettings>::default()
                    .load(path.join("best_model_half.pth").into(), &device)
                    .expect("Failed to decode state");

            // Save the model record to a file.
            let recorder = BinBytesRecorder::<HalfPrecisionSettings>::default();

            let bin = recorder
                .record(record, ())
                .expect("Failed to save model record");

            // Save into the OUT_DIR directory so that the model can be loaded by the
            let file_path = Path::new("ckpt");
            std::fs::create_dir_all(&file_path).expect("Failed to create directory");
            let file_path = file_path.join(format!("{}.bin", name));
            std::fs::write(&file_path, bin).expect("Failed to write file");
        }
    }
}
