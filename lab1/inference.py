import os
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator


def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list):
    # TODO plot training and testing accuracy curve
	x = range(1, len(train_acc_list)+1)
	plt.clf()
	plt.title('Accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy(%)')
	plt.plot(x, train_acc_list, '-ob', x, val_acc_list, '-r') 
	plt.legend(["train accuracy", "test accuracy"], loc ="lower right")
	plt.savefig('accuracy.png')
	print("train_acc_list")
	print(train_acc_list)
	print("val_acc_list")
	print(val_acc_list)

def plot_f1_score(f1_score_list):
    # TODO plot testing f1 score curve
	x = range(1, len(train_acc_list)+1)
	plt.clf()
	plt.title('F1 score')
	plt.xlabel('epoch')
	plt.ylabel('F1 score')
	plt.plot(x, f1_score_list, '-ob')
	plt.savefig('f1_score.png')
	print("f1_score_list")
	print(f1_score_list)

def plot_confusion_matrix(confusion_matrix):
    # TODO plot confusion matrix
	plt.clf()
	df = pd.DataFrame(confusion_matrix, columns=['Predicted Normal', 'Predicted Pneumonia'],
						index=['Actual Normal', 'Actual Pneumonia'])
	sns.heatmap(df, annot=True)
	plt.title('Confusion Matrix')
	plt.savefig('confusion_matrix.png')
	print("confusion matrix")
	print(confusion_matrix)

def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model_name', type=str, required=False, default='ResNet18')
    parser.add_argument('--model_path', type=str, required=False, default='best_model_weights.pt')

    # for training
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    if args.model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif args.model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)

    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    model.load_state_dict(torch.load(os.path.join(args.model_path)))

    # evaluate
    val_acc, f1_score, c_matrix = test(test_loader, model)

    # plot
    plot_confusion_matrix(c_matrix)
