import os
import numpy as np
import click
import yaml
import torch 
from Models.model3att import get_model
from numpy import asarray
from tqdm import tqdm
from PIL import Image

foreground_threshold = 0.25 # percentage of pixels of val 255 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    patch = np.where(patch >.5, 1, 0) #/ 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

@click.command()
@click.option('-p', '--config_path', default='configs/config.yml',type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    model = get_model()
    model.eval()
    load_path = config.get('load_pth', "")
    if load_path != "":
        try:
            model.load_state_dict(torch.load(load_path))
            print("Restored weights from pretrained model.")
        except Exception as e:
            print(e)
            print("Error occurred when restoring weights.")
            exit()
    base_dir = "data/test_set_images/ethz-cil-2023"
    image_filenames = [os.path.join(base_dir, name) for name in os.listdir(base_dir)]
    with open("submission.csv", 'w') as f:
        f.write('id,prediction\n')

        for img in tqdm(image_filenames):
            img_nr = img[-7:-4]

            x = asarray(Image.open(img).convert('RGB'))
            
            x = torch.tensor(x)
            x = (x.to(torch.float)-128)/128
            x = x.unsqueeze(0)
            x = x.permute(0,3,1,2)

            x = model(x)
            x = torch.squeeze(x).detach().numpy()
            patch_size = 16
            
            for j in range(0, x.shape[1], patch_size):
                for i in range(0, x.shape[0], patch_size):
                    patch = x[i:i + patch_size, j:j + patch_size]
                    label = patch_to_label(patch)
                    f.write("{}_{}_{},{}\n".format(img_nr, j,i,label))

if __name__ == '__main__':
    main()