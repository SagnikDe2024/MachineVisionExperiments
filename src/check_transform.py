import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image, ImageReadMode, write_jpeg, write_png
from torchvision.transforms.v2.functional import pad_image, rotate, InterpolationMode, crop

below = 0.01


def conv_test(image_file: str):
    img = decode_image(image_file, mode=ImageReadMode.RGB)

    # transform = v2.Compose([v2.Resize((512, 512))])
    # resized = transform(img)
    perform_fft_and_save(img)
    # perform_fft_and_save(resized)


def filter_before_writing(img):
    img_f = img.to(dtype=torch.float)
    img_f_mi = img_f.min()
    img_f_ma = img_f.max()
    if img_f_mi < 0 or img_f_ma > 255:
        nine_five = torch.quantile(img_f, 1 - below)
        five = torch.quantile(img_f, below)
        clamped = torch.clamp(img_f, five, nine_five)
        img_f_scaled = (clamped - five) * 255 / (nine_five - five)
        print(f'Filtered -> {below} -> {five} , {1 - below} -> {nine_five} ')
        return img_f_scaled
    else:
        return img_f


def perform_fft_and_save(img):
    summed = img.sum()
    print(f'size = {img.shape} and sum = {summed}')

    pic_fft = torch.fft.fft2(img)
    pic_fft_abs = pic_fft.abs()
    pic_fft_angle = pic_fft.angle()
    trim_below = torch.quantile(pic_fft_abs, below)
    trim_above = torch.quantile(pic_fft_abs, 1 - below)
    clamped_abs = torch.clamp(pic_fft_abs, trim_below, trim_above)
    pic_fft_clamped = torch.multiply(clamped_abs, torch.exp(1j * pic_fft_angle))

    print(f'min = {pic_fft_abs.min()} , median -> {pic_fft_abs.median()} , max -> {pic_fft_abs.max()} ')
    print(f'Clamped {below} = {trim_below} , {1 - below} -> {trim_above}')

    # clamped_log = torch.log(clamped_abs)
    hist = torch.histc(pic_fft_abs, 256, min=trim_below, max=trim_above)
    bins = np.arange(0, 256)
    plt.plot(bins, hist)
    plt.show()

    fft_real = pic_fft_clamped.real
    fft_imag = pic_fft_clamped.imag

    converted = filter_before_writing(torch.fft.ifft2(pic_fft_clamped))
    converted_re = filter_before_writing(torch.fft.ifft2(fft_real))
    converted_im = filter_before_writing(torch.abs(torch.fft.ifft2(fft_imag).imag))
    write_jpeg(converted.to(dtype=torch.uint8), "../out/converted.jpg")
    write_jpeg(converted_re.to(dtype=torch.uint8), "../out/converted_re.jpg")
    write_jpeg(converted_im.to(dtype=torch.uint8), "../out/converted_im.jpg")


def gen_fourier_image(size, angle):
    radius = size // 2
    max_radius = radius * (2 ** 0.5)
    diameter = radius * 2
    max_value = diameter ** 2 * 127
    p = np.log(max_value) / np.log(max_radius)
    # pic2d = np.zeros((diameter, diameter), dtype=np.complex64)
    phases = np.random.rand(diameter, diameter)
    pic2d = np.sin(phases * 2 * np.pi) + 1j * np.cos(phases * 2 * np.pi)
    for y in range(diameter):
        for x in range(diameter):
            r2 = ((x - radius) ** 2 + (y - radius) ** 2)
            if r2 > 0:
                abs_val = max_value * (r2 ** (-p / 2))
                # f_component = abs_val*phase
                pic2d[y, x] *= abs_val
    image_with_fft = torch.tensor(pic2d).unsqueeze(0)
    conv = torch.fft.fft2(image_with_fft)
    print(f'Shape = {conv.shape}')

    # convreal = conv.real
    convreal = torch.abs(conv)
    max_value_r = convreal.max()
    min_value_r = convreal.min()
    med = torch.median(convreal).item()
    # Maximum value and median assuming min_value_r is zero
    max_v = max_value_r - min_value_r
    med_v = med - min_value_r
    # Assume that pixel value p with range [0,255]
    # now A 255^n = max_v and A 128^n = med_v
    # then we have n = log2(max_v/med_v) and A = max_v/(255^n)
    n = np.log2(max_v / med_v)
    A = max_v / (255 ** n)
    # So now p = (y/A)^(1/n) for pixel value
    normed = torch.float_power(torch.divide(torch.subtract(convreal, min_value_r), A), 1 / n)

    print(f'Max = {max_value_r}, Min = {min_value_r} and median = {med}')
    write_jpeg(normed.to(dtype=torch.uint8), "../out/ifft_n.jpg")
    return normed





if __name__ == '__main__':
    # result = gen_fourier_image(512,np.pi/6)
    # print(result)
    img = decode_image("../data/normal_pic.jpg", mode=ImageReadMode.RGB)
    img = img.to(dtype=torch.float32)
    img = img/255
    crop_size = 256
    padding = crop_size//4
    img = crop(img,400,1244 ,  crop_size,crop_size)
    increased_img = pad_image(img,padding=padding)

    increased_img_45 = rotate(increased_img,45,InterpolationMode.BILINEAR)
    increased_img_135 = rotate(increased_img,-45,InterpolationMode.BILINEAR)
    rot_x_diff = torch.diff(increased_img_45,dim=2)
    rot_y_diff = torch.diff(increased_img_135,dim=2)

    x_diff = rotate(rot_x_diff,-45,InterpolationMode.BILINEAR)
    y_diff = rotate(rot_y_diff,45,InterpolationMode.BILINEAR)
    x_diff_abs = torch.abs(x_diff)
    y_diff_abs = torch.abs(y_diff)
    max_diff = (x_diff_abs + y_diff_abs - 2*(x_diff_abs*y_diff_abs))/(2-x_diff_abs-y_diff_abs)
    cropped_diff = crop(max_diff,padding-1,padding-1, crop_size+2,crop_size+2)*255










    # img = rotate(img,45,InterpolationMode.BILINEAR)
    # # img = resize(img,[256,256])
    # complex_image = convert_rgb_to_complex(img)
    # undiff_complex = convert_complex_to_rgb(complex_image).squeeze(0)
    # write_png(undiff_complex.to(dtype=torch.uint8), "../out/complex_crop.png")
    # diff = find_complex_diff(complex_image)
    # diff_abs_max = diff.abs().max()
    # diff_scaled = diff*255 / diff_abs_max
    # rgb_image = convert_complex_to_rgb(diff_scaled).squeeze(0)
    # # rgb_rotated_back = rotate(rgb_image,-45,InterpolationMode.BILINEAR)

    write_png(cropped_diff.to(dtype=torch.uint8), "../out/complex_pipe_max_crop.png")




    # pdf = calculate_pdf(resized)
    # log_pdf = np.log2(pdf,out = np.zeros_like(pdf, dtype=np.float32), where = (pdf != 0))
    # multiplied = np.multiply(-log_pdf, pdf)
    # entropy = np.sum(multiplied)
    # print(f'Total entropy = {entropy}')
    # plt.contourf(log_pdf[1])
    # plt.axis('square')
    # plt.show()

    # conv_test("../data/normal_pic.jpg")
