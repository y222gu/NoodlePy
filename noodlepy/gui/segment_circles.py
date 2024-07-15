import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt


def onclick(event,img,i=0):
    x, y = event.xdata, event.ydata
    print('i,x,y', i, x, y)
    click_point.append([x, y])
    output.append_stdout(f'Clicked coordinates: ({x}, {y})\n')
    predictor.set_image(img)

    input_point = np.array(click_point)
    input_label = np.arange(1, input_point.shape[0] + 1)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    img=cv2.applyColorMap(img, cv2.COLORMAP_JET)
    mask=np.array((masks[0],masks[0],masks[0])).transpose((1,2,0))

    cv2.imshow('mask', (255*mask).astype('uint8'))
    cv2.imwrite(f'./mask/{i}.bmp', (255*mask).astype('uint8'))
    np.save(f'./mask/mask.npy', mask_all)
    cv2.imshow('img', img.astype('uint8'))
    masked_img=img*mask
    cv2.imshow('masked_img', masked_img.astype('uint8'))
    cv2.waitKey(10)

def connect_callback(extra_param1, extra_param2):
    def onclick_with_params(event):
        onclick(event, extra_param1, extra_param2)
    return onclick_with_params

ckpt_path = "/Users/yifeigu/Downloads/sam_vit_b_01ec64.pth"
sam_checkpoint = "/Users/yifeigu/Downloads/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

import numpy as np
import torch
ri_path = '/media/renzhihe/data/Katz/logs/musc3D/musc3Dz0-1_fs180_bs500_b2c580_zz20dz15dx45_c2f_med01norm_L12_nocs_tvxy0e6_maxri1_lr1e2/701/RI.npy'
ri_path = './ri3d/RI_41600.npy'
image_stack = np.load(ri_path)[:,:,:]
mask_all =np.zeros_like(image_stack)

for i in range(image_stack.shape[2]):
    print(i)
    img = image_stack[:, :, i]
    img=cv2.imread('test_raman.png',0)
    img = img / np.max(img) * 255

    img = np.stack([img] * 3, axis=-1).astype(np.uint8)

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from IPython.display import display
    import ipywidgets as widgets

    image = img

    output = widgets.Output()
    click_point=[]

    fig, ax = plt.subplots()
    ax.imshow(image)

    cid = fig.canvas.mpl_connect('button_press_event', connect_callback(img,i))
    print(output)
    plt.show()

    display(output)

    print('layer:',)



