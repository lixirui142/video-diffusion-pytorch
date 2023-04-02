import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm

data_root = "data"
# aniheads = osp.join(data_root, "anihead_gif", "1665379386678_konosuba_s1_sample_no_10.mkv.gif")
# heads = Image.open(aniheads)
# print(np.array(heads).shape)


moving_MNIST = osp.join(data_root, "mnist_test_seq.npy")
mM = np.load(moving_MNIST)
print(mM.shape)
mM = mM.transpose(1, 0, 2, 3)

for i, vdo in enumerate(tqdm(mM)):
    vdo = [Image.fromarray(image) for image in vdo]
    vdo[0].save(osp.join(data_root, "moving_MNIST" ,f"{i}.gif"), save_all=True, append_images=vdo[1:], duration=1000/24, loop=0)
