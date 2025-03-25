import numpy as np
import torch
import torch.nn.functional as fn
from numpy import ndarray, dtype
from torchvision.io import write_png


class DiffuseForwardPass(object):
    def __init__(self):
        self.noise = 0.1


def calculate_pdf(pic: torch.tensor):
    print(f'shape: {pic.shape}')
    x_shift_kernel = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.half)
    x_shift_kernel = x_shift_kernel.repeat(3,1,1)
    x_shift_kernel = x_shift_kernel.unsqueeze(0)
    # x_shift_kernel.repeat(4,3)
    y_shift_kernel = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.half).unsqueeze(0).unsqueeze(0)
    pic_un = pic.unsqueeze(0).to(dtype=torch.half)
    (ch, h, w) = pic.shape
    pic_uny = pic_un.permute(0, 1, 3, 2)

    pic_shift_x = torch.concat((torch.zeros(ch,h,1),pic[:,:,1:]),2)
    pic_shift_y = torch.concat((torch.zeros(ch,1,w),pic[:,1:,:]),1)
    pic_shift_z = torch.concat((torch.zeros(1,h,w),pic[1:,:,:]),0)
    shft_x = pic - pic_shift_x
    shft_y = pic - pic_shift_y
    shft_z = pic - pic_shift_z


    # pic_x = fn.conv2d(pic_un, x_shift_kernel, bias=None, padding=1)
    # # pic_y = fn.conv2d(pic_un, y_shift_kernel, bias=None, padding=1)
    # pic_y_un = fn.conv2d(pic_uny, x_shift_kernel, bias=None, padding=1)
    # pic_y = pic_y_un.permute(0, 1, 3, 2)

    if ch > 1:
        # pic_unz = pic_un.permute(0, 3, 2, 1)
        # pic_z_un = fn.conv2d(pic_unz, x_shift_kernel, bias=None, padding=1)
        # pic_z = pic_z_un.permute(0, 3, 2, 1)
        # pic_x = pic_x.squeeze()
        # pic_y = pic_y.squeeze()
        # pic_z = pic_z.squeeze()
        pic_x = shft_x
        pic_y = shft_y
        pic_z = shft_z
        pdf: ndarray[tuple[int, int, int], dtype[np.float32]] = np.zeros([511, 511, 511], dtype=np.float32)
        for px in range(w):
            for py in range(h):
                for pz in range(ch):
                    xv = int(pic_x[pz, py, px])
                    yv = int(pic_y[pz, py, px])
                    zv = int(pic_z[pz, py, px])
                    pdf[zv + 255, yv + 255, xv + 255] += 1
        pdf = pdf / np.sum(pdf)
        return pdf
    else:
        # pic_x = pic_x.squeeze().squeeze()
        # pic_y = pic_y.squeeze().squeeze()
        pdf: ndarray[tuple[int, int], dtype[np.float32]] = np.zeros([511, 511], dtype=np.float32)
        # for px in range(w):
        #     for py in range(h):
        #         xv = int(pic_x[py, px])
        #         yv = int(pic_y[py, px])
        #         pdf[yv + 255, xv + 255] += 1
        pdf = pdf / np.sum(pdf)
        return pdf

# def get_gradients(pic : torch.tensor):
#     print(f'shape: {pic.shape}')
#     x_shift_kernel = torch.tensor([[0, 0, 0], [-1/2, 1, -1/2], [0, 0, 0]], dtype=torch.half)

def find_complex_diff(pic: torch.tensor):
    (n,h,w) = pic.shape
    xdiff = torch.diff(pic, dim=2, prepend=torch.zeros(n,h,1))



    write_png(convert_complex_to_rgb(xdiff).squeeze(0).to(dtype=torch.uint8), "../../out/complex_xdiff.png")
    ydiff = torch.diff(pic, dim=1, prepend=torch.zeros(n,1,w))
    write_png(convert_complex_to_rgb(ydiff).squeeze(0).to(dtype=torch.uint8), "../../out/complex_ydiff.png")

    npx = np.array(xdiff)
    npy = np.array(ydiff)
    print(f'{pic}')
    print(f'npx: {npx}, npy: {npy}')
    r_diff = torch.where(xdiff.real.abs() >  ydiff.imag.abs(),xdiff.real,ydiff.imag)
    i_diff = torch.where(xdiff.imag.abs() > ydiff.real.abs(),xdiff.imag,ydiff.real)
    complex_diff = r_diff + 1j*i_diff
    return complex_diff



def convert_rgb_to_complex(image):
    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.168736, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]], dtype=torch.float32)
    # Reshape the image to (N, C, H, W) if it's not already
    if image.ndim == 3:
        image = image.unsqueeze(0)
    # Permute the image to (N, H, W, C)
    image = image.permute(0, 2, 3, 1)
    ycbcr_image = torch.trunc(torch.tensordot(image.to(dtype=torch.float32), transform_matrix, dims=([3], [1])))
    # Permute the image to (C, N, H, W)
    # ycbcr_image_ch_form = ycbcr_image.permute(3, 0, 1, 2)
    luma_only = ycbcr_image[:,:,:,0]
    cb_cr_only = ycbcr_image[:, :, :, 1:]
    complex_cb_cr = torch.view_as_complex(cb_cr_only.contiguous())
    print(f'luma = {luma_only}')
    print(f'cb_cr = {complex_cb_cr}')
    hue_angle = complex_cb_cr.angle()
    real = torch.trunc(luma_only * torch.cos(hue_angle))
    imag = torch.trunc(luma_only * torch.sin(hue_angle))
    complex_image = real + 1j*imag
    return complex_image


def convert_complex_to_rgb(image):
    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.168736, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]], dtype=torch.float32)
    inverse_transform_matrix = torch.inverse(transform_matrix)
    # Reshape the image to (N, H, W, C) if it's not already
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image_luma = image.abs()
    print(f' luma_max: {image_luma.max()} , luma_min: {image_luma.min()} ')
    image_hue_angle = image.angle()
    cb = torch.cos(image_hue_angle)*(1-image_luma/255)*image_luma/(1.772*2)
    cr = torch.sin(image_hue_angle)*(1-image_luma/255)*image_luma/(1.402*2)
    print(f'cb_max = {cb.max()} , cb_min = {cb.min()} , cr = {cr.max()} , cr = {cr.min()}')


    ycbcr_image = torch.cat((image_luma, cb, cr), dim=1)
    # Permute to (N, H, W, C)
    ycbcr_image = ycbcr_image.permute(0, 2, 3, 1)
    rgb_image_raw = torch.tensordot(ycbcr_image, inverse_transform_matrix, dims=([3], [1]))

    # Permute the image to (N,C,H,W)
    rgb_image = rgb_image_raw.permute(0, 3, 1, 2)

    print(f' r_max = {rgb_image[:,0,:,:].max()} , r_min = {rgb_image[:,0,:,:].min()} ')
    print(f' g_max = {rgb_image[:,1,:,:].max()} , g_min = {rgb_image[:,1,:,:].min()} ')
    print(f' b_max = {rgb_image[:,2,:,:].max()} , b_min = {rgb_image[:,2,:,:].min()} ')

    return rgb_image

    









