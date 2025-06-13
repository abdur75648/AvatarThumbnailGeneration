# Avatar Thumbnail Generation

## Libraries required
```bash
pip install opencv-python numpy gdown mediapipe torch torchvision
```

## Ckpt Download
```bash
gdown --id 1KhcCNrSW_lPRcMSWMQRk802qRQIC2zRf -O modnet_photographic_portrait_matting.ckpt
```

## Usage
```python
python3.11 single_avatar_thumbnail_script.py --thumbnail_image_path_rgb sample_input.png
```

