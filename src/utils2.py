from csv import writer
import pandas as pd


class Profiler():
    def __init__(self, name):
        columns = ['train_acc_mean', 'train_loss_mean', 'kldl_train_mean', 'klds_train_mean', 'rec_train_mean', 'ce_train_mean',
                   'train_acc_std', 'train_loss_std', 'kldl_train_std', 'klds_train_std', 'rec_train_std', 'ce_train_std',
                   'valid_acc_mean', 'valid_loss_mean', 'kldl_valid_mean', 'klds_valid_mean', 'rec_valid_mean', 'ce_valid_mean']
        #columns = columns.append(additional)
        df = pd.DataFrame(columns=columns)
        self.path = '/home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/' + name + '.csv'
        df.to_csv(self.path, index=False)

    def log(self, row):
        with open(self.path, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(row)

# /home/nfs/anujsingh/meta_lrng/files/learning_to_meta-learn/logs/
# /home/anuj/Desktop/Work/TU_Delft/research/implement/learning_to_meta-learn/logs
