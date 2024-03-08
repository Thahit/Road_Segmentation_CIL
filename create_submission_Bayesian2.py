import os
import numpy as np
import click
import yaml
import torch 
from Models.model1_BAYESIAN import get_model_two_step
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
    uncert_model, model = get_model_two_step()
    model.train()
    load_path = config.get('load_pth', "")
    if load_path != "":
        try:
            model.load_state_dict(torch.load(load_path))
            print("Restored weights from pretrained model.")
        except Exception as e:
            print(e)
            print("Error occurred when restoring weights.")
            exit()
        try:
            uncert_model.load_state_dict(torch.load(os.path.dirname(os.path.normpath(load_path)) + '/model1.pth'))
            print("Restored weights from pretrained model.")
        except Exception as e:
            print(e)
            print("Error occurred when restoring weights.")
            exit()
    base_dir = "data/test_set_images/ethz-cil-2023"
    image_filenames = [os.path.join(base_dir, name) for name in os.listdir(base_dir)]



    print("create final predictions")
    with open("submission.csv", 'w') as f:
        f.write('id,prediction\n')
        with torch.no_grad():
            for img in tqdm(image_filenames):
                img_nr = img[-7:-4]

                x = asarray(Image.open(img).convert('RGB'))
                x = torch.tensor(x)
                x = (x.to(torch.float)-128)/128
                x = x.unsqueeze(0)
                x = x.permute(0,3,1,2)
                # create uncertainties:
                results = []
                for _ in range(20):
                    pred = uncert_model(x)
                    results.append(pred)
                result = torch.var(torch.stack(results), dim=0)
                # max var of binomial = .25
                result *= 100 # [0, 250] 
                result = torch.round(result)
                x = torch.cat((x, result), 1)                

                out = model(x)
                for _ in range(19):
                    out += model(x)
                out = torch.squeeze(out).detach().numpy()/20
                #img = (out*255).astype(np.uint8)
                #img = Image.fromarray(img, mode='L')
                #img.save('submit/' + img_nr + '.png')
                patch_size = 16
                
                for j in range(0, out.shape[1], patch_size):
                    for i in range(0, out.shape[0], patch_size):
                        patch = out[i:i + patch_size, j:j + patch_size]
                        label = patch_to_label(patch)
                        f.write("{}_{}_{},{}\n".format(img_nr, j,i,label))

if __name__ == '__main__':
    main()