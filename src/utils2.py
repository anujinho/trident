import os
import torch
from csv import writer
import pandas as pd


class Profiler():
    def __init__(self, name, experiment):
        # columns = ['train_acc_mean', 'train_loss_mean', 'kldl_train_mean', 'klds_train_mean', 'rec_train_mean', 'ce_train_mean',
        #            'train_acc_std', 'train_loss_std', 'kldl_train_std', 'klds_train_std', 'rec_train_std', 'ce_train_std',
        #            'valid_acc_mean', 'valid_loss_mean', 'kldl_valid_mean', 'klds_valid_mean', 'rec_valid_mean', 'ce_valid_mean']
        columns = ['task', 'accuracy', 'ELBO', 'Label_KL', 'Style_KL', 'Reconst_Loss', 'CE_Loss']
        #columns = columns.append(additional)
        df = pd.DataFrame(columns=columns)
        self.path = '/home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/' + name + '/' + experiment
        os.makedirs(self.path, mode=0o777)

        self.path_train = self.path + '/' + 'train.csv'
        self.path_valid = self.path + '/' + 'valid.csv'
        self.path_test = self.path + '/' + 'test.csv'
        df.to_csv(self.path_train, index=False)
        df.to_csv(self.path_valid, index=False)
        df.to_csv(self.path_test, index=False)


    def log_csv(self, row, mode):
        d = {'train': self.path_train, 'valid': self.path_valid, 'test': self.path_test}
        with open(d[mode], 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            if mode == 'train':
                csv_writer.writerows(row)
            else:
                csv_writer.writerow(row)

    def log_imgs(self, images, epoch, mode):
        self.path_imgs = self.path + '/' + 'images_epoch-' + epoch + '_' + mode + '.pt'
        torch.save(images, self.path_imgs)


# /home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/
# /home/anuj/Desktop/Work/TU_Delft/research/implement/learning_to_meta-learn/logs/
