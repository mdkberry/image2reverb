# Image2Reverb

## LATEST INFORMATION:

Adaptation underway to see if it can be used to work with Comfyui for image or ideally video ambience when applying audio to a scene clip created in comfyui. I would recommend not installing this version at this stage while it is being worked on. I will post updates here if it becomes something I think is going to be useful and useable - mdkberry (August 2025)

---

#### Image2Reverb: Cross-Modal Reverb Impulse Response Synthesis
Nikhil Singh, Jeff Mentch, Jerry Ng, Matthew Beveridge, Iddo Drori

[__Project Page__](https://web.media.mit.edu/~nsingh1/image2reverb/)

Code for the ICCV 2021 paper [[arXiv]](https://arxiv.org/abs/2103.14201). Image2Reverb is a method for generating audio impulse responses, to simulate the acoustic reverberation of a given environment, from a 2D image of it.

![](webpage/src/splash.png)


## Dependencies

**Model/Data:**

* PyTorch>=1.7.0
* PyTorch Lightning
* torchvision
* torchaudio
* librosa
* PyRoomAcoustics
* PIL

**Eval/Preprocessing:**

* PySoundfile
* SciPy
* Scikit-Learn
* python-acoustics
* google-images-download
* matplotlib

## Image Requirements

* Images should be 512x512 pixels for best results
* Supported formats: JPG, PNG, BMP, TIFF

## Usage

### Running on a Single Image

To generate an impulse response from a single image, you can use the provided `run_single_image.py` script:

```bash
python run_single_image.py --image_path path/to/your/image.jpg --output_dir ./results
```

This script will:
1. Resize your image to 512x512 pixels if needed
2. Create a temporary dataset structure
3. Run the model on your image
4. Save the results in the specified output directory
5. Clean up temporary files

### Required Pre-trained Models

The required pre-trained models should be placed in the `models` folder:
1. Places365 ResNet50 model: `models/resnet50_places365.pth.tar`
2. Monodepth2 models: `models/mono_640x192/` folder containing `encoder.pth` and `depth.pth`
3. Image2Reverb checkpoint: `models/model.ckpt`

If you haven't already downloaded these models, you can get them from:
1. Places365 ResNet50 model: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar
2. Monodepth2 models: From https://github.com/nianticlabs/monodepth2
3. Image2Reverb checkpoint: https://media.mit.edu/~nsingh1/image2reverb/model.ckpt

## Resources

[Model Checkpoint](https://media.mit.edu/~nsingh1/image2reverb/model.ckpt)


## Code Acknowlegdements

We borrow and adapt code snippets from [GANSynth](https://github.com/magenta/magenta/tree/master/magenta/models/gansynth) (and [this](https://github.com/ss12f32v/GANsynth-pytorch) PyTorch re-implementation), additional snippets from [this](https://github.com/shanexn/pytorch-pggan) PGGAN implementation, [monodepth2](https://github.com/nianticlabs/monodepth2), [this](https://github.com/jacobgil/pytorch-grad-cam) GradCAM implementation, and more.

## Citation

If you find the code, data, or models useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Singh_2021_ICCV,
    author    = {Singh, Nikhil and Mentch, Jeff and Ng, Jerry and Beveridge, Matthew and Drori, Iddo},
    title     = {Image2Reverb: Cross-Modal Reverb Impulse Response Synthesis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {286-295}
}
```
