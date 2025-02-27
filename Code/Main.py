import torch
from torch.utils.data import DataLoader
from TransformerLearner import TransformerLearner
from loss import MIL
from Avenue_dataset import Normal_Loader, Anomaly_Loader
import os
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from input_dataset import * #batch_size = 32
train_loss_history = []
auc_history = []

parser = argparse.ArgumentParser(description='PyTorch MIL Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--w', default=0.001, type=float, help='weight_decay')
parser.add_argument('--modality', default='TWO', type=str, help='modality')
parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
parser.add_argument('--FFC', '-r', action='store_true', help='FFC')
args = parser.parse_args()

best_auc = 0

normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)

anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality)
anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)

# Define a dynamic batch size based on the dataset size
batch_size = min(len(normal_train_dataset), len(anomaly_train_dataset))

normal_train_loader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=False)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=batch_size, shuffle=False)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=batch_size, shuffle=False)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.FFC:
    model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
else:
    model = TransformerLearner(input_dim=args.input_dim, dropout=args.drop).to(device)

optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL

# Train until a stopping criterion is met
epoch = 0
max_epochs = 100  # You can adjust this as needed

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0

    # Get the minimum batch size between normal and anomaly inputs
    min_batch_size = min(len(normal_train_loader), len(anomaly_train_loader))

    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        print(f"Batch {batch_idx}:")
        print("Normal inputs shape:", normal_inputs.shape)
        print("Anomaly inputs shape:", anomaly_inputs.shape)
        
        # Use the smaller batch size
        batch_size = min(min_batch_size, normal_inputs.shape[0], anomaly_inputs.shape[0])

        inputs = torch.cat([anomaly_inputs[:batch_size], normal_inputs[:batch_size]], dim=1)
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)  # Use the dynamically calculated batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    average_train_loss = train_loss / min_batch_size  # Use min_batch_size
    train_loss_history.append(average_train_loss)
    print('loss =', average_train_loss)
    scheduler.step()


def test_abnormal(epoch):
    model.eval()
    global best_auc
    auc = 0

    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            print("",frames)
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]


            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0]//16, 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)

        average_auc = auc / len(anomaly_test_loader)
        auc_history.append(average_auc)
        print('auc = ', average_auc)
        if best_auc < average_auc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = average_auc

# Train for a dynamic number of epochs until convergence based on validation metric
while epoch < max_epochs:
    train(epoch)
    test_abnormal(epoch)

    # Check for a stopping criterion (e.g., based on AUC improvement)
    if len(auc_history) >= 2 and auc_history[-1] <= auc_history[-2]:
        break
    
    epoch += 1

# Plotting code
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(auc_history)), auc_history, label='AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()
