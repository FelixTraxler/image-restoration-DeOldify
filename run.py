from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
import torch
torch.backends.cudnn.benchmark=True
from deoldify.visualize import *

colorizer = get_image_colorizer(artistic=False)
# colorizer.plot_transformed_image("test_images/image.png", render_factor=35, compare=True)

#NOTE:  Max is 45 with 11GB video cards. 35 is a good default
render_factor=35
#NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
# source_url='https://upload.wikimedia.org/wikipedia/commons/e/e4/Raceland_Louisiana_Beer_Drinkers_Russell_Lee.jpg'
source_path = '../input_128/temp.png'

results_dir = Path('../output_images')

# if source_url is not None:
#     colorizer.plot_transformed_image_from_url(url=source_url, path=source_path, render_factor=render_factor, compare=True)
# else:
colorizer.plot_transformed_image(path=source_path, results_dir=results_dir, render_factor=render_factor, compare=True)

print(f"Result saved to {results_dir / 'temp_DeOldify.png'}")