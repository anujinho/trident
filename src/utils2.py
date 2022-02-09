import os
import torch
from csv import writer
import pandas as pd


class Profiler():
    def __init__(self, name, experiment, args):
        self.columns = ['task', 'accuracy', 'ELBO', 'Label_KL', 'Style_KL', 'Reconst_Loss', 'CE_Loss']
        #columns = columns.append(additional)
        df = pd.DataFrame(columns=self.columns)
        self.args = args
        self.path = '/home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/' + name + '/' + experiment
        os.makedirs(self.path, mode=0o777)

        self.path_train = self.path + '/' + 'train.csv'
        self.path_valid = self.path + '/' + 'valid.csv'
        self.path_test = self.path + '/' + 'test.csv'
        self.path_preds = self.path + '/' + 'preds.csv'
        self.path_test_all = self.path + '/' + 'test_all.csv'
        df.to_csv(self.path_train, index=False)
        df.to_csv(self.path_valid, index=False)
        df.to_csv(self.path_test, index=False)
        df.to_csv(self.path_preds, index=False)
        df.to_csv(self.path_test_all, index=False)



    def log_csv(self, row, mode):
        d = {'train': self.path_train, 'valid': self.path_valid, 'test': self.path_test, 'test_all': self.path_test_all, 'preds': self.path_preds}
        with open(d[mode], 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            if self.args.backbone[0] == True:
                csv_writer.writerows(row)
            else:
                if mode == 'train':
                    csv_writer.writerows(row)
                else:
                    csv_writer.writerow(row)


    def log_data(self, data, epoch, mode1, mode2):
        self.path_data = self.path + '/' + mode1 + '_epoch-' + str(epoch) + '_' + mode2 + '.pt'
        torch.save(data, self.path_data)
    
    def log_model(self, model, opt, epoch):
        self.path_model = self.path + '/' + 'model_' + str(epoch) + '.pt'
        self.path_opt = self.path + '/' + 'opt_' + str(epoch) + '.pt'
        torch.save(model, self.path_model)
        torch.save(opt.state_dict(), self.path_opt)


# /home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/
# /home/anuj/Desktop/Work/TU_Delft/research/implement/learning_to_meta-learn/logs/
