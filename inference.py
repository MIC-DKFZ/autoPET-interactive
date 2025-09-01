from pathlib import Path

from glob import glob
import SimpleITK
import numpy as np
import json
import os

import torch

from nnunetv2.inference.autopet_predictor import autoPETPredictor
from nnunetv2.training.dataloading.nnInteractive_clicks import PointInteraction_stub

'''
docker run --rm --platform=linux/amd64 --network none --gpus all --volume /path/to/your/test/cases/input:/input --volume /path/to/your/test/cases/output:/output autopet4
'''

# LOCAL TEST
CT_INPUT_PATH = Path("/input/images/ct/")
PET_INPUT_PATH = Path("/input/images/pet/")
CLICKS_INPUT_PATH = Path("/input/lesion-clicks.json")
OUTPUT_PATH = Path("/output/images/tumor-lesion-segmentation/")

RESOURCE_PATH = Path("path/to/checkpoint/")  # Change this to the path where you have stored the nnUNet model checkpoints
POINT_WIDTH = 2

def run():
    # Read the input
    # Read the input
    input_array_ct, spacing, direction, origin, uuid = load_image_file_as_array(
        location=CT_INPUT_PATH,
    )
    input_array_pet, _, _, _, _ = load_image_file_as_array(
        location=PET_INPUT_PATH,
    )

    clicks = load_json(CLICKS_INPUT_PATH)
    
    input_array = np.stack([input_array_ct, input_array_pet])
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    ############# Lines You can change ###########
    # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['nnUNet_compile'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    

    spacing_for_nnunet=list(spacing)[::-1]
    props = {
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
        # are returned x,y,z but spacing is returned z,y,x. Duh.
        'spacing': spacing_for_nnunet
    }

    predictor = autoPETPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    predictor.initialize_from_trained_model_folder(RESOURCE_PATH, 
                                                    use_folds=(0,1,2,3,4), 
                                                    checkpoint_name='checkpoint_final.pth')
    
    input_array = input_array.astype(np.half)

    import time
    start = time.time()
    ret = predictor.predict_single_npy_array(input_array, props, clicks, POINT_WIDTH, None, None, False)
    print("Time taken for prediction: ", time.time() - start)


    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(input_array[0], name='input')
    # viewer.add_image(np.flip(input_array, axis=3)[0], name='input_flipped')
    # viewer.add_image(ret, name='output')
    # viewer.add_image(np.flip(ret, axis=2), name='output_flipped')
    # napari.run()

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH,
        array=ret,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
        uuid=uuid,
    )
    print('Saved.')
    return 0


def sparse_to_dense_point_nnInteractive_inference(points: dict[str, np.ndarray], shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    point_interaction = PointInteraction_stub(point_radius=sigma, use_distance_transform=True)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            if label == 'tumor':
                pos_clicks = point_interaction.place_point(coord, pos_clicks, binarize=False)
            elif label == 'background':
                neg_clicks = point_interaction.place_point(coord, neg_clicks, binarize=False)
            else:
                raise ValueError(f"Unknown label {label} in click json")
    return pos_clicks, neg_clicks


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r") as json_file:
        return json.load(json_file)


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    uuid = os.path.splitext(os.path.basename(input_files[0]))[0]
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin, uuid


def write_array_as_image_file(*, location, array, spacing, origin, direction, uuid):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / Path(uuid + suffix),
        useCompression=True,
    )


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(torch.__version__)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())