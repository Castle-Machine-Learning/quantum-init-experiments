import os
import time
import zmq  # the message protocol
import pickle
import threading
import numpy as np
from qiskit import Aer, IBMQ
from qrandom import get_backend, get_array
from datetime import datetime


def _create_message(string_lst: list) -> bytes:
    """Create a message for the server.

    Args:
        string_lst (list): A list of messages.

    Returns:
        bytes: The transmossion bytes. 
    """    
    time = str(datetime.now())
    message = time 
    for string in string_lst:
        message += '+' + string
    return message.encode('utf8')




class QuantumRndProvidingServer(object):

    def __init__(self,
                 bind = "tcp://*:5555",
                 backend='least_busy',
                 storage_size: float =1e5,
                 storage_file_name: str ='storage.pkl',
                 ):
        """Set up the random number providing server.

        Args:
            bind (str, optional): Where to bind the socket to. Defaults to "tcp://*:5555".
            backend (str, optional): Where to get the numbers from.
                                     I.e. 'armonk', 'athens' ,'least_busy' or 'simulator', .
                                     Defaults to 'least_busy'.
            storage_size (float, optional): Size of the internal buffer for incoming requests.
                                     Defaults to 1e5.
            storage_file_name (str, optional): The name of the file where old numbers are backed up.
                                               Defaults to 'storage.pkl'.
        """                 
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(bind)
        self.storage_size = storage_size

        if backend == 'simulator':
            self.backend = Aer.get_backend('qasm_simulator')
        elif backend == 'athens':
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            self.backend = provider.get_backend('ibmq_athens')
        elif backend == 'armonk':
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            self.backend = provider.get_backend('ibmq_armonk')
        else:
            # get the least busy backend
            self.backend = get_backend()

        # storage for used random numbers.
        self.storage_file_name = storage_file_name

        self.array_lock = threading.Lock()
        self.refill_thread = threading.Thread()
        self.listen_thread = threading.Thread()
        self.random_array = np.array([]).reshape(0)



    def run(self):
        """ Obtain new random numbers and serve incoming requests.
            The implementation to avoid getting stuck while waiting
            for quantum numbers or incoming requests. """
        if len(self.random_array) < self.storage_size:
            # We don't have enogh randomness in store.
            # Get more.
            if not self.refill_thread.is_alive():
                self.refill_thread = threading.Thread(
                    target=self._request_new_random,
                    name='refill_thread')
                self.refill_thread.start()
                print('fill_thread started.')
        if not self.listen_thread.is_alive():
            self.listen_thread = threading.Thread(
                target=self._listen,
                name='listen_thread'
            )
            self.listen_thread.start()

    def _listen(self):
        message = self.socket.recv()
        now = datetime.now()
        print(str(now) + ": " + "Received request: %s" % message)
        self._process_message(message)

    def _process_message(self, message: bytes):
        """ Respond to incoming messages.

        Args:
            message (bytes): The read out of the message.
        """        
        message = message.decode()
        splits = message.split('+')
        now = datetime.now()

        if splits[1] == "ping":
            #  Send reply back to client
            now = datetime.now()
            print(str(now) + ": " + 'sending reply to ping.')
            self.socket.send(_create_message(["ping recieved"]))
        elif splits[1] == "rnd":
            print(splits[2])
            rnd_array = self._get_random(int(splits[2]))
            print(str(now) + ": " + 'sending ' + str(rnd_array))
            self.socket.send(rnd_array.tobytes())

    def _get_random(self, number_no: int) -> np.array:
        """ Get uniformly distributed random numbers
            from the internal storage.

        Args:
            number_no (int): The number of random numbers.

        Returns:
            np.array: The uniformly distributed numbers,
        """        
        if len(self.random_array) >= number_no:
            with self.array_lock:
                requested_no = self.random_array[:number_no]
                # delete the returned numbers.
                self.random_array = self.random_array[number_no:]
            return requested_no
        else:
            self._request_new_random()
            return self._get_random(number_no)

    def _request_new_random(self):
        """ Request new random numbers from self.backend. """        
        qbits = self.backend.configuration().n_qubits
        max_shots = self.backend.configuration().max_shots
        ints = int(qbits/16. * max_shots)
        new_numbers = get_array((ints), qbits,
                                   self.backend)
        with self.array_lock:
            self.random_array = np.concatenate([self.random_array,
                                                new_numbers])
        # store the new numbers.
        if os.path.isfile(self.storage_file_name):
            storage_array = pickle.load( open( self.storage_file_name, "rb" ) )
            storage_array = np.concatenate([storage_array,
                                            new_numbers])
        else:
            storage_array = new_numbers
      
        pickle.dump( storage_array, open( self.storage_file_name, "wb" ) )        

    def load_from_storage(self, filename: str = 'storage.pkl'):
        """Load previously stored numbers.

        Args:
            filename (str, optional): The path to the stored numbers.
                                      Defaults to 'storage.pkl'.
        """
        storage_array = pickle.load( open( filename, "rb" ) )
        with self.array_lock:
            self.random_array = storage_array


# Run the Server
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Set up an FCMLQ quantum provider.')
    parser.add_argument('--backend', type=str, default='armonk',
                        help='[str] Choose the quantum backend \
                              i.e. simulator, athens, armonk, or least_busy. \
                              Defaults to least_busy .')
    parser.add_argument('--port', type=int, default=5555,
                        help='[int] The port where the server should listen.\
                              Defaults to: 5555')
    parser.add_argument('--load-storage', action='store_true', default=False,
                        help='Load stashed random numbers.')

    args = parser.parse_args()
    print(args)

    server = QuantumRndProvidingServer(bind="tcp://*:" + str(args.port),
                                       backend=args.backend)
    if args.load_storage:
        server.load_from_storage()
    while True:
        #  Wait for next request from client
        server.run()

