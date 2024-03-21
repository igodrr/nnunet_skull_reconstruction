import os
import SimpleITK as sitk
import pandas as pd
import numpy as np

def rle_encode(img):
    pixels = img.flatten()
    if np.all(pixels == 0):
        return '0'
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# folder_path = '/home/data/lrd/zgp/MedNeXt/submit/mm_S5_ep1000_all'
folder_path = "/home/data1/zhangzr22/skull_3d/nnUNet/results/2d/0"
nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
nii_files.sort()
data_rows = []

current_id = 1


for file in nii_files:
    sitk_image = sitk.ReadImage(file)

    image_data = sitk.GetArrayFromImage(sitk_image)

    assert image_data.shape[0] == 150


    for i in range(image_data.shape[0]):
        slice_data = image_data[i, :, :]
        rle_encoded = rle_encode(slice_data)
        data_rows.append({'id': current_id, 'predicted': rle_encoded})
        current_id += 1


df = pd.DataFrame(data_rows)

df.to_csv('submission_zzr_2d_205.csv', index=False)