from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import pywt
import cv2
from PIL import Image



dirpath="64.png"
img_PIL = Image.open(dirpath)
img_PIL = np.array(img_PIL)
data=np.array(img_PIL).astype(np.float32)
data=data.reshape(64,64,3)
four=np.array(data)
data1 = cv2.resize(data, (10,10))
mask11 = np.array(data1).reshape(3,10,10).astype(np.float32)
data2 = cv2.resize(data, (32,32))
mask2 = np.array(data2).reshape(3,32,32).astype(np.float32)



def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def setwaterMark(Img):
    c = pywt.wavedec2(Img,'db2',level=3)
    [cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = c
    d = pywt.wavedec2(mask11, 'db2', level=1)
    [ca1, (ch1, cv1, cd1)] = d
    e = pywt.wavedec2(mask2, 'db2', level=2)
    [cb1, (chb1, cvb1, cdb1),(chb2, cvb2, cdb2)] = e
    a1 = 0
    a2 = 0.8
    a3 = 0.8
    a4 = 0.8
    cl = cl + ca1 * a1
    cH3 = cH3 + ch1 * a2
    cV3 =  cv1 * a3
    cD3 =  cd1 * a4
    cH2 = cH2+ chb1 * a2
    cV2 =  cvb1 * a3
    cD2 = cdb1 * a4
    newImg = pywt.waverec2([cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)],'db2')
    newImg = np.array(newImg,np.float32)
    return newImg
def setwaterMark2(Img):
    c = pywt.wavedec2(Img,'db2',level=3)
    [cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = c
    d = pywt.wavedec2(mask11, 'db2', level=1)
    [ca1, (ch1, cv1, cd1)] = d
    e = pywt.wavedec2(mask2, 'db2', level=2)
    [cb1, (chb1, cvb1, cdb1),(chb2, cvb2, cdb2)] = e
    a1 = 0
    a2 = 1
    a3 = 1
    a4 = 1

    cl = cl + ca1 * a1
    cH3 = cH3 + ch1 * a2
    cV3 =  cv1 * a3
    cD3 = cd1 * a4

    cH2 = cH2*(1-a2) + chb1 * a2
    cV2 = cvb1 * a3
    cD2 =  cdb1 * a4

    newImg = pywt.waverec2([cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)],'db2')
    newImg = np.array(newImg,np.float32)
    return newImg

def DUBA(data):
    m = np.array(data).reshape(3, 32, 32).astype(np.float32)
    m2 = np.array(data).reshape(32, 32, 3).astype(np.float32)
    new=setwaterMark(m)
    img_np = new.reshape(32,32,3).astype(np.float32)
    fft_trg_cp = np.fft.fft2(img_np, axes=(-3, -2))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-3, -2)).astype(np.float32)

    fft_source_cp = np.fft.fft2(m2, axes=(-3, -2))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-3, -2))
    amp_source_shift = amp_source_shift
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-3, -2))
    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_target)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-3, -2))
    local_in_trg = np.real(local_in_trg).reshape(32,32,3).astype(np.float32)
    local_in_trg[m2>=190]=m2[m2>=190]
    local_in_trg[m2<=60]=m2[m2<=60]

    return img_np

def DUBA2(data):
    m = np.array(data).reshape(3, 32, 32).astype(np.float32)
    m2 = np.array(data).reshape(32, 32, 3).astype(np.float32)
    new=setwaterMark2(m)
    img_np = new.reshape(32,32,3).astype(np.float32)

    fft_trg_cp = np.fft.fft2(img_np, axes=(-3, -2))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-3, -2)).astype(np.float32)

    fft_source_cp = np.fft.fft2(m2, axes=(-3, -2))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-3, -2))
    amp_source_shift = amp_source_shift
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-3, -2))
    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_target)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-3, -2))
    local_in_trg = np.real(local_in_trg).reshape(32,32,3).astype(np.float32)
    local_in_trg[m2>=230]=m2[m2>=230]
    local_in_trg[m2<=20]=m2[m2<=20]

    return img_np






def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='dataset', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='dataset', train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='dataset', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return train_clean_loader

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger','duba']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'duba':
            img = self._dubaTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)
        return img_

    def _dubaTrigger(self, img, width, height, distance, trig_w, trig_h):

        img_ = DUBA(img)
        img_ = np.clip((img_).astype('uint8'), 0, 255)
        
        
        return img_
