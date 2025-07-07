import cv2
import numpy as np
import time
import torch
import gc

def load_cube_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    lut_data = []
    lut_size = 0
    for line in lines:
        line = line.strip()
        if line.startswith('LUT_3D_SIZE'):
            lut_size = int(line.split()[1])
        elif line == '' or line.startswith('#') or line.startswith('TITLE') or line.startswith('DOMAIN'):
            continue
        else:
            lut_data.append([float(x) for x in line.split()])

    lut = np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3))
    return lut

def apply_3d_lut_gpu(image_np, lut_np, device='cuda'):

    img = torch.from_numpy(image_np).float().div(255.0).to(device)  
    lut = torch.from_numpy(lut_np).float().to(device)       

    H, W = img.shape[:2]
    size = lut.shape[0]
    img_lut = img * (size - 1)
    i = torch.floor(img_lut).long()
    f = img_lut - i
    i = torch.clamp(i, 0, size - 2)

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    ix, iy, iz = i[..., 0], i[..., 1], i[..., 2]

    out = torch.zeros_like(img)

    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                w = ((1 - dx) + fx * (2 * dx - 1)) * \
                    ((1 - dy) + fy * (2 * dy - 1)) * \
                    ((1 - dz) + fz * (2 * dz - 1))

                x = ix + dx
                y = iy + dy
                z = iz + dz

                # gather LUT values (batch indexing)
                c = lut[x, y, z]  # shape: (H, W, 3)

                out += w.unsqueeze(-1) * c

    out = (out * 255.0).clamp(0, 255).byte()
    res = out.cpu().numpy()
    
    del img, out, lut
    gc.collect()
    torch.cuda.empty_cache()

    return res





if __name__ == "__main__":

    img = cv2.imread("./test_lut/cat.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lut = load_cube_file("./test_lut/1.cube")
    for _ in range(10):
        t = time.time()
        img_filtered  = apply_3d_lut_gpu(img_rgb, lut)
        print(time.time() - t)
    cv2.imwrite("./test_lut/cat_res.jpg", img_filtered)