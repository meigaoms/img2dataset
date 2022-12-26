import os 
import cv2
import numpy as np
import math
import os
import random
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt


'''
    process a part pic of laion-hr_ori

'''

def filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    # radius = min(fshift.shape[0],fshift.shape[1])
    # radius = int(radius_ratio * radius / 2)
    yradius = int(radius_ratio * template.shape[1] / 2)
    xradius = int(radius_ratio * template.shape[0] / 2)
    if len(fshift.shape) == 3:
        cv2.ellipse(template, (crow, ccol), (xradius,yradius), 0, 0, 360, (1, 1, 1),  -1)
        # cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        # cv2.circle(template, (crow, ccol), radius, 1, -1)
        cv2.ellipse(template, (crow, ccol), (xradius,yradius), 0, 0, 360, 1,  -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift
 
def filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    yradius = int(radius_ratio * filter_img.shape[1] / 2)
    xradius = int(radius_ratio * filter_img.shape[0] / 2)
    if len(filter_img.shape) == 3:
        cv2.ellipse(filter_img, (crow, col), (xradius,yradius), 0, 0, 360, (0, 0, 0),  -1)
        # cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.ellipse(filter_img, (crow, col), (xradius,yradius), 0, 0, 360, 0,  -1)
        # cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    cnt_array = np.where(filter_img==0,1,0)
    lowcnt = int(np.sum(cnt_array))
    highcnt = int(np.sum(np.ones(fshift.shape, np.uint8))) - lowcnt
    return filter_img * fshift,lowcnt ,highcnt
 

def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg

def cal_en(fimg):
    w ,h = fimg.shape
    enr = 0
    for i in range(w) :
        for j in range(h):
            enr += abs(fimg[i,j])**2
    return enr

def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
    # 获取低频和高频部分
    hight_parts_fshift,lowcont,highcont = filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
    # print(hight_parts_fshift)
    ratio = cal_en(hight_parts_fshift)/(cal_en(low_parts_fshift)+cal_en(hight_parts_fshift))
    img_new_low, img_new_high = None,None
    return img_new_low, img_new_high,ratio

def randomcrop(img):
    # pathch_size = 512
    H,W = img.shape
    if H != W:
        pathch_size = min(H,W)
        # pathch_size = min(pathch_size,patch)
    else:
        pathch_size = H
    rnd_h = random.randint(0, max(0, H - pathch_size))
    rnd_w = random.randint(0, max(0, W - pathch_size))
    img_L = img[rnd_h:rnd_h + pathch_size, rnd_w:rnd_w + pathch_size]

    return img_L,pathch_size


def entropy(img_str):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0

    img_str.seek(0)
    img_content = bytearray(img_str.read())
    file_bytes = np.asarray(img_content, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#     image = cv2.imread(img_path,0)
    h,w = image.shape
    pixel = h*w
    # with open(img_path, "rb") as f:
    size = len(img_content)
    bpp = float(size*8) / pixel
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    noise_sigma = estimate_sigma(image, average_sigmas=True)
    img_patch,patch_size = randomcrop(image)
    low_freq_part_img, high_freq_part_img ,ratio= get_low_high_f(img_patch, radius_ratio=0.5) 
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_np = np.array(gray_img)
    hist = get_hist(img, False)
    is_select = select_with_th(hist, 0.7, range_width=5)
    return (h,w),bpp,res,noise_sigma,is_select,ratio

def get_hist(image_np, isshow=False):
    hist_ = cv2.calcHist([image_np], [0], None, [256], [0,256])
    hist = []
    for i in hist_:
        hist.append(i[0])
    if isshow:
        plt.bar(range(len(hist)), hist)
        plt.show()
    return hist

def select_with_th(hist, th, range_width=0):
    # cal the ratio of each pixel
    hist_np = np.array(hist)
    number = sum(hist_np)
    ratio_hist_np = hist_np / number
    if ratio_hist_np.max() >= th:
        return False
    else:
        if range_width > 0:
            max_id = np.argmax(hist_np)
            id_l = max(0, max_id-range_width)
            id_r = min(256, max_id+range_width)
            select_ratio = ratio_hist_np[id_l:id_r+1].sum()
            if select_ratio >= th:
                return False
            else:
                return True

def imquality_highresolution(img_str,reason_printer=False):
    '''
    input :
        filepath
    output :
        True for selected picture
        False for filter outs.
    Filter by:
        1.resolution 
        2.compress ratio
        3.image entropy
        4.high frequency ratio
        5.noise level
        [(hxw),pixel,bpp,entropy,ratio,category]
        select step 1:
            a)h,w>1000,
            b)ratio>0.008
            c)bbp>6
            d)entroy>6
            e)20 >noise_sigma>1e-2
    '''
    try:
        (h,w),bpp,res,noise_sigma,hist,ratio = entropy(img_str)
        if not hist:
            if reason_printer:
                print('for rgb value center on one value > 0.7')
            return False
        elif h<1000 or w<1000 :
            if reason_printer:
                print('for low resolution')
            return False
        elif bpp <=1 :
            if reason_printer:
                print('for highly compressed ')
            return False
        elif res <= 4.9183 :
            if reason_printer:
                print('for low  image entropy ')
            return False
        elif ratio <= 0.005 :
            if reason_printer:
                print('for low  high frequency ratio ')
            return False
        elif noise_sigma <= 1e-2 or noise_sigma > 35:
            if reason_printer:
                print('for high noise level  ')
            return False
        else:
            return True
    except:
        print('Error in process pic:')


def is_ellipse(mask):
    points = np.column_stack(np.where(mask.transpose()))
    hull = cv2.convexHull(points)
    ((cx, cy), (ew, eh), angle) = cv2.fitEllipse(hull)
    zero_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    eclipse_mask = cv2.ellipse(zero_mask, (int(cx), int(cy)), (int(ew/2), int(eh/2)), angle, 0, 360, 255, -1)
    ell_area = np.sum(eclipse_mask>0)
    all_fg_area = np.sum(mask)
    ell_fg_area = np.sum(mask*(eclipse_mask>0))
    
    if ell_fg_area / float(all_fg_area) > 0.9 and  ell_fg_area / float(ell_area) > 0.9:
        return True
    else:
        return False

def qualified_rgba(img_str):
    try:
        img_str.seek(0)
        file_bytes = np.asarray(bytearray(img_str.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        # Discard non-rgba images
        if len(img.shape) == 2 or img.shape[2]<4:
            return False
        # Discard small images
        if min(img.shape[0], img.shape[1])<500:
            return False
        mask = img[:,:,3]
        binary_mask = mask > 128
        trans_mask = np.logical_and(mask>0.05*255, mask<0.95*255)
        fg_ys, fg_xs = np.nonzero(binary_mask)
        # Discard all black mask
        if fg_ys.size == 0:
            return False
        # ??
        if np.sum(trans_mask)/float(np.sum(binary_mask)) > 0.5:
            return False
        min_x, max_x = np.min(fg_xs), np.max(fg_xs)
        min_y, max_y = np.min(fg_ys), np.max(fg_ys)

        bbox_area = (max_x-min_x+1)*(max_y-min_y+1)
        fg_area = np.sum(binary_mask[min_y:max_y+1, min_x:max_x+1])

        fg_h, fg_w = max_y - min_y + 1, max_x - min_x +1
        fg_ratio = float(fg_area) / (bbox_area + 0.01)
        if fg_area < 16 or fg_ratio > 0.88:
            return False
        is_ell = is_ellipse(binary_mask)
        if is_ell:
            return False
        return True
    except:
        return False