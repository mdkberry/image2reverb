# Image2Reverb Dataset Structure

This folder contains the dataset structure for training and testing the Image2Reverb model.

## Folder Structure

- `train_A/` - Training images (input)
- `train_B/` - Training audio impulse responses (output)
- `val_A/` - Validation images (input)
- `val_B/` - Validation audio impulse responses (output)
- `test_A/` - Test images (input)
- `test_B/` - Test audio impulse responses (output)

## Data Format

- Images should be in common formats (JPG, PNG, etc.)
- Audio files should be in WAV format
- Paired files (image and corresponding audio) should have the same base filename
- Images should be 512x512 pixels for best results (resized automatically if needed)

## Usage

When training or testing the model, use the `--dataset` parameter pointing to this directory:

```bash
python train.py --dataset ./datasets/image2reverb
python test.py --dataset ./datasets/image2reverb
