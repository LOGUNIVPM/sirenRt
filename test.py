import argparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchaudio
from dataset import *
from model import *
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


def test_accuracy(model, device, dataloader, batch_size):
    test_loss = 0.0
    class_correct = list(0. for i in range(params.num_classes))
    class_total = list(0. for i in range(params.num_classes))
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if len(labels.data) != batch_size:
                    break

                # Forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(inputs)

                # Loss computation
                loss = loss_fn(outputs, labels)

                # Test loss update
                test_loss += loss.item()*inputs.size(0)

                # Conversion of output probabilities to predicted class
                _, pred = torch.max(outputs, 1)

                # Predictions to true label comparison
                correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

                # Computation of test accuracy for each class
                for i in range(batch_size):
                    target = labels.data[i]
                    class_correct[target] += correct.item()
                    class_total[target] += 1
                t.update()

            # Compute and print average test loss
            test_loss = test_loss/len(dataloader.dataset)
            print("Test Loss: {:.3f}\n".format(test_loss))

            for i in range(params.num_classes):
                print("Test Accuracy of %5s: %.2f%% (%2d/%2d)" %
                    (str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))

            print("\nTest Accuracy (Overall): %.2f%% (%2d/%2d)" %
                (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


def test_auprc(model, device, dataloader):
    auc_list = []
    test_pred = []
    test_true = []
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for jdx, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                y_pred = cnn(inputs)
                y_pred = y_pred[:, 1].view(-1).detach()
                test_pred.append(y_pred.cpu().detach().numpy())
                test_true.append(labels.numpy())
                t.update()

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(test_true, test_pred)
            auc_tmp = sklearn.metrics.auc(recall, precision)
            auc_list.append(auc_tmp)

        auc = np.mean(auc_list)
        print("Area under precision recall curve: {:.3f}".format(auc))

        fig, ax = plt.subplots()
        pr_display = sklearn.metrics.PrecisionRecallDisplay(precision, recall)
        pr_display.plot(ax=ax)
        plt.show()


if __name__ == "__main__":

    # Link to config file
    args = parser.parse_args()
    params = Params(args.config_path)

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
        print("Device: {}".format(device))
        print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
    else:
        device = torch.device("cpu")
        print("Device: {}".format(device))

    # Instantiate our dataset object and create the dataloader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=params.sample_rate,
        n_fft=params.fft_size,
        win_length=params.window_length,
        hop_length=params.hop_size,
        n_mels=params.mels_number
    )

    test_set = SirenSynthDataset(params.test_annotations_file,
                                 params.test_audio_dir,
                                 mel_spectrogram,
                                 params.sample_rate,
                                 params.num_samples,
                                 device)

    print(f"There are {len(test_set)} samples in the test set.")
    signal, label = test_set[0]

    # Construct the model and assign it to device
    cnn = SirenCNN()
    summary(cnn.to(device), (1, 128, 51))

    # Specify the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # Load the best saved model
    checkpoint = os.path.join(params.checkpoint_dir, "model_best.pt")
    saved_model = load_checkpoint(checkpoint, cnn, optimizer, parallel=False)

    # Create Pytorch data samplers and loaders
    test_dataloader = create_test_dataloader(test_set, params.test_batch_size)

    # Test with accuracy as metric
    if params.metrics == "accuracy":
        test_accuracy(cnn, device, test_dataloader, params.test_batch_size)

    # Test with auprc as metric
    elif params.metrics == "auprc":
        test_auprc(cnn, device, test_dataloader)
