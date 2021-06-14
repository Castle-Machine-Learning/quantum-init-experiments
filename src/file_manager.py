import pickle


class RandomnessFileManager(object):

    def __init__(self, filename, init_pos=0):
        """ Create a manager for numpy randomness files.

        Args:
            filename (str): A path to a pickeled numpy array with
                random numbers.
            init_pos (int): Initial position in file to avoid reusing
                old numbers.
        """
        self.filename = filename
        self.pos = init_pos
        self.storage_array = pickle.load(open(filename, "rb"))

    def request_rnd(self, number_count):
        """ Get random numbers from the logfile.

        Args:
            number_count (int): The requested amount of random numbers.

        Returns:
            [np.array]: A numpy array with the desired numbers.
        """
        print('pos', self.pos)
        numbers = self.storage_array[self.pos:(self.pos+number_count)]
        self.pos = self.pos + number_count
        return numbers


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    anu = RandomnessFileManager('./numbers/storage-5-ANU_3May2012_100MB-unshuffled-32bit-160421.pkl')
    for i in range(10):
        sample = anu.request_rnd(1000)
        plt.plot(sample, '.')
    plt.show()
    print('done')
