import os
import shutil
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder, ImageFolder
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
import pywt
import paddle.vision.transforms as T
from backdoor import WaNet,FIBA,Blend,BadNets,DUBA,DUBA2
mean,std = ([0.5, 0.5,0.5],[0.5, 0.5,0.5])
mean = list(map(lambda x:x*255,mean))
std = list(map(lambda x:x*255,std))
#prepare data
train_parameters = {
    "class_dim": 200,  #classes
    "target_path":"data",
    'train_image_dir': 'data/train',
    'eval_image_dir': 'data/val',
    'test_image_dir': 'data/val',
}

class MyDataset(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()

        train_image_dir = train_parameters['train_image_dir']
        eval_image_dir = train_parameters['eval_image_dir']
        test_image_dir = train_parameters['eval_image_dir']

        transform_train = Compose([Resize(size=(64, 64))])
        transform_eval = Compose([Resize(size=(64, 64))])
        train_data_folder = DatasetFolder(train_image_dir, transform=transform_train)
        eval_data_folder = DatasetFolder(eval_image_dir, transform=transform_eval)
        test_data_folder = ImageFolder(test_image_dir, transform=transform_eval)
        self.mode = mode
        if self.mode == 'train':
            self.data = train_data_folder
        elif self.mode == 'eval':
            self.data = eval_data_folder
        elif self.mode == 'test':
            self.data = test_data_folder
        print(mode, len(self.data))

    def __getitem__(self, index):

        if self.mode == 'train':
            if index % 10 == 1:
                data = np.array(self.data[index][0])
                img_np=DUBA(data)
                data = img_np
                label = 1

            else:
                data = np.array(self.data[index][0])

                m = np.array(data).reshape(64, 64, 3).astype(np.float32)
                img_np = T.Transpose()(m)
                img_np = T.Normalize(mean, std)(img_np)
                data = img_np
                label = self.data[index][1]
        elif self.mode == 'eval':
            data = np.array(self.data[index][0])

            m = np.array(data).reshape(64, 64, 3).astype(np.float32)
            img_np = T.Transpose()(m)
            img_np = T.Normalize(mean, std)(img_np)
            data = img_np
            label = self.data[index][1]

        else:
            data = np.array(self.data[index][0])
            m = np.array(data).reshape(3, 64, 64).astype(np.float32)
            img_np = DUBA2(data)
            data = img_np
            data = img_np
            label = 1

        return data, label

    def __len__(self):

        return len(self.data)


train_dataset_2 = MyDataset(mode='train')
val_dataset_2 = MyDataset(mode='eval')
test_dataset_2 = MyDataset(mode='test')

from paddle.vision.models import resnet50
model = resnet50(pretrained=True,num_classes=200)
model = paddle.Model(model)
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')

model.prepare(optimizer=paddle.optimizer.Adam(
              learning_rate=0.001,
              parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

model.fit(train_dataset_2,
          epochs=10,
          batch_size=32,
          callbacks=callback,
          verbose=1)
model.evaluate(test_dataset_2,batch_size=32)
model.evaluate(val_dataset_2,batch_size=32)