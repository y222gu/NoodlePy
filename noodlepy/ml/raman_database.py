# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset

class SyntheticRamanFileDataset(Dataset):
    def __init__(self, filepath):
        """
            Assumes that your spectra are saved as files
            Advantages:  You can distribute your training data to others
            Disadvantages:  You have to keep track of your training sets
        """

        # TODO: open the csv file with panda and store it in self.spectra_df
        # TODO: initialize the common wavelength range (ie. it could be the first wavelength in the file round up to three decimals)
        # TODO: interpolate all spectra to the common wavelength range (scipy.interpolate.interp1d(method='bilinear'))""))
        # potentially overwrite the original spectra with the interpolated ones

    def __len__(self):
        return len(self.spectra_df.shape[0])

    def __getitem__(self, idx):
        # TODO: Get the idx-th row
        # TODO: Create two random augmentation dictionaries
        # TODO: apply the augmentation to the idx-th spectrum
        return augmentation1, augmentation2
    

class SyntheticRamanOnTheFlyDataset(Dataset):
    def __init__(self, number_of_spectra):
        """
            Creates the spectra on call
            Advantage:  No additional creation of files
            Disadvantage:   You need to make sure that your random numpy and torch seeds are 
                            properly intilizalized or you will get a different answer every time

                            see https://numpy.org/doc/stable/reference/random/generator.html
                            see https://pytorch.org/docs/stable/notes/randomness.html
                            see https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
        """
        # TODO: Initialize random spectra parameters e.g. valid ranges for each metabolite self.dictionary = {'metabolite 1':[low,high]}        
        #                                            e.g. open a config file with ranges for each metabolite
        self.number_of_spectra = number_of_spectra

    def __len__(self):
        return self.number_of_spectra

    def __getitem__(self, idx):
        # TODO: Draw a random set of metabolite compositions out of the ranges
        # TODO: Turn them into a speectrum
        # TODO: Create two random augmentation dictionaries
        # TODO: apply the augmentation to the spectrum        
        return augmentation1, augmentation2