#!/usr/bin/env python
import os
import glob
import optparse

import cv2
import torch
import numpy as np

from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# - Fixed seed (PyTorch, Numpy, Python)
# - What is accuracy, precicion, recall?
# - Model checkpoints
# - Use nn.Sequential in Model.forward()
# - Have a description on the final YOLO layers, how would we implement it?
#   Tranforming(Rotation, Scaling)? 
# - Learn about regularization. Implement in model, save as different checkpoints.


## Things we should need in the final result
# - Save error and verification error to file so we can inspect it as a graph
# - 


class CatsAndDogsDataset(Dataset):
    def __init__(self, f_name, transform):
        self.transform = transform
        self.dogcat_list = []

        cats_paths = os.path.join(f_name, "cats", "*.jpg")
        cats = glob.glob(cats_paths)
        dog_paths = os.path.join(f_name, "dogs", "*.jpg")
        dogs = glob.glob(dog_paths)
        ## 0 = dogs
        ## 1 = cats
        dog_constant = torch.zeros(1)
        cat_constant = torch.ones(1)

        if torch.cuda.is_available():
            dog_constant = dog_constant.cuda()
            cat_constant = cat_constant.cuda()

        for path in dogs:
            item = (dog_constant, path)
            self.dogcat_list.append(item)

        for path in cats:
            item = (cat_constant, path)
            self.dogcat_list.append(item)

    def __len__(self):
        return len(self.dogcat_list)

    def __getitem__(self, idx):
        filename = self.dogcat_list[idx][1]
        classCategory = self.dogcat_list[idx][0]
        im = Image.open(filename)
        if self.transform:
            im = self.transform(im)
        return im.view(-1), classCategory, filename


class CatsAndDogsModel(nn.Module):
    def __init__(self, image_size, out_dim):
        super(CatsAndDogsModel, self).__init__()
        in_dim = image_size[0] * image_size[1]

        self.layer = nn.Sequential(
            nn.Linear(in_dim, 20000),
            nn.Linear(20000, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer(x)


def serialize_model(model, epoch, loss, validate_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'validate_loss': validate_loss,
    }, 'model.{}.pt'.format(epoch))


def load_model_from_file(model, path):
    "Returns the saved epoch of the model"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch']


def train_model(model, epochs, dataset, test_dataset, save_every=10, start_epoch=0):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Max mem: {torch.cuda.max_memory_allocated()}, allocated: {torch.cuda.memory_allocated()}")

    model.train(True)

    learning_rate = 0.001
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    try:
        inter_epoch_loss = 0
        for epoch in range(epochs):
            print("epoch #{}".format(epoch+start_epoch+1))
            print("Herro people")
            running_loss = 0
            for images, labels, path in dataloader:
                optimiser.zero_grad()

                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                    #print(f"device: {torch.cuda.get_device_name(images.get_device())}")


                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                inter_epoch_loss += loss.item()
            else:
                print(f"Total epoch loss: {running_loss/len(dataloader)}")

            print(f"Max mem: {torch.cuda.max_memory_allocated()}, allocated: {torch.cuda.memory_allocated()}")

            if epoch % save_every == 0:
                validate_loss = validate_model(model, test_dataset)
                serialize_model(model, epoch+start_epoch, inter_epoch_loss, validate_loss)
                inter_epoch_loss = 0
    except KeyboardInterrupt:
        pass

    model.train(False)


def validate_model(model, dataset):
    "Returns the loss of a single validation epoch"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train(False)

    criterion = nn.MSELoss()

    total = 0

    cats_total = 0
    cats_total_tp = 0
    cats_total_fp = 0

    dogs_total = 0
    dogs_total_tp = 0
    dogs_total_fp = 0

    running_loss = 0
    for images, labels, _ in dataloader:

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()


        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()

        results = output > 0.5
        results = list(zip(results, labels))

        for c in results:
            tp = 0
            fp = 0

            if c[0] == c[1].byte():
                tp += 1
            elif c[0] != c[1].byte():
                fp += 1

            if c[0] == 0:
                dogs_total += 1
                dogs_total_tp += tp
                dogs_total_tp += fp
            elif c[0] == 1:
                cats_total += 1
                cats_total_tp += tp
                cats_total_fp += fp

        total += len(results)

        running_loss += loss.item()

    # test_model is called from withing train_model, as such we should return to training mode    

    model.train(True)

    cats_precision = cats_total_tp / (cats_total_tp + cats_total_fp)
    dogs_precision = dogs_total_tp / (dogs_total_tp + dogs_total_fp)

    cats_recall = cats_total_tp / (cats_total)
    dogs_recall = dogs_total_tp / (dogs_total)

    print("cats: ", cats_precision, ", dogs: ", dogs_precision)
    print("cats: ", cats_recall, ", dogs:", dogs_recall)

    return running_loss


if __name__ == "__main__":
    torch.manual_seed(69)
    np.random.seed(69)

    print(f"Using cuda: {torch.cuda.is_available()}")

    image_size = (100, 100)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_dataset = CatsAndDogsDataset('dataset/training_set/', transform)
    test_dataset = CatsAndDogsDataset('dataset/test_set/', transform)


    model = CatsAndDogsModel(image_size, 1)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

#    start_epoch = load_model_from_file(model, 'first_model/model.2300.pt')
    start_epoch = 0
    train_model(model, 1, training_dataset, test_dataset, save_every=5, start_epoch=start_epoch)
    #validate_model(model, test_dataset, device)

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    total_tested = 0
    total_correct = 0

    for img, label, path in dataloader:
        output = model(img.cuda())
        cv_img = cv2.imread(path[0])
        #cv_img = img.detach().reshape((100, 100)).numpy()

        cv_img = cv2.resize(cv_img, (400, 400))
#        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGBA)
        cv2.putText(cv_img, f"truth: {'cat' if label > 0.5 else 'dog'}, {label.item()}", (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))

        correct = bool(output.item() > 0.5) == bool(label.item() > 0.5)

        total_tested += 1
        total_correct += 1 if correct else 0

        if correct:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(cv_img, f"guess: {'cat' if output.item() > 0.5 else 'dog'}, {output.item():.5f}", (0, 60), cv2.FONT_HERSHEY_DUPLEX, 1, color)
        cv2.putText(cv_img, f"{total_correct}/{total_tested}, {float(total_correct)/float(total_tested)*100.0:.2f}%", (0, 90), cv2.FONT_HERSHEY_DUPLEX, 1, color)
        cv2.imshow("random", cv_img)
        cv2.waitKey(400)

    print(f"{total_correct}/{total_tested}, {float(total_correct)/float(total_tested)*100.0:.2f}%")

    # Find Learning rate - 0.001
    # Find optmiser      - SGD
    # Find loss function - MSE(Mean squared error), BCE(binary cross entropy)

    # for (images, cats) in dataloader:
    #     for i in range(len(cats)):
    #         out = model.forward(images[i])
    #         print("model out: {}, is_cat: {}".format(float(out), float(cats[i])))
