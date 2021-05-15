from csv import writer
import pandas as pd


class Profiler():
    def __init__(self, name):
        df = pd.DataFrame(columns=['train_acc_mean', 'train_acc_std', 'train_loss_mean',
                          'train_loss_std', 'valid_acc_mean', 'valid_acc_std', 'valid_loss_mean', 'valid_loss_std'])
        self.path = '../logs/' + name + '.csv'
        df.to_csv(self.path, index=False)

    def log(self, row):
        with open(self.path, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(row)
