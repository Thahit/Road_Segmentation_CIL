import os
import numpy as np
import torch 
from numpy import asarray
from tqdm import tqdm
from PIL import Image

#import all the models you want to use
import Models.attention as att
import Models.model1 as m1
import Models.model3att2 as m3a2


foreground_threshold = 0.25 # percentage of pixels of val 255 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    patch = np.where(patch >.5, 1, 0) #/ 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def main():
    models = []
    # unfortunatley due to the models being always in different 
    # folders etc, this part can only be done manually

    #repeat this part for each model:
    models.append(att.get_model())

    load_path = "output/att/model_checkpoint/best.pth"
    try:
        models[-1].load_state_dict(torch.load(load_path))
        print("Restored weights from pretrained model.")
    except Exception as e:
        print(e)
        print("Error occurred when restoring weights.")
        exit()
    
    models.append(m1.get_model())

    load_path = "output/m1/model_checkpoint/best.pth"
    try:
        models[-1].load_state_dict(torch.load(load_path))
        print("Restored weights from pretrained model.")
    except Exception as e:
        print(e)
        print("Error occurred when restoring weights.")
        exit()
    

    models.append(m3a2.get_model())

    load_path = "output/m3a2/model_checkpoint/best.pth"
    try:
        models[-1].load_state_dict(torch.load(load_path))
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

            out = models[0](x)
            for m in models[1:]:#attention! this would not work as intended if mayesian netowrks are used
                out += m(x)
            out = torch.squeeze(out).detach().numpy() / len(models)

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