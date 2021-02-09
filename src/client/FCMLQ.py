import zmq
import numpy as np
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


def test_ping(address: str = "tcp://localhost:5555"):
    """ Test if server is available. """
    context = zmq.Context()
    #  Socket to talk to server
    print("Connecting to server.")
    socket = context.socket(zmq.REQ)
    socket.connect(address)

    # send
    message = _create_message(['ping'])
    now = datetime.now()
    print(str(now) + " Sending request ,", message)
    socket.send(message)

    #  Get the reply.
    message = socket.recv()
    now = datetime.now()
    print(str(now) + " Received reply [ %s ]" % ( message))


def request_rnd(random_number_total: int,
                address: str = "tcp://localhost:5555") -> np.array:
    """ Request uniformly distributed quantum random numbers 
        from U[0, 1].

    Args:
        random_number_total (int): The number of random numbers required.
        address (str, optional): The location where the Quantum randomness
            server is runnung. Defaults to "tcp://localhost:5555".

    Returns:
        np.array: A vector array of shape [random_number_total].
    """    
    context = zmq.Context()

    #  Socket to talk to server
    print("Requesting.", str(random_number_total), ' numbers')
    socket = context.socket(zmq.REQ)
    socket.connect(address)

    # send
    message = _create_message(['rnd', str(random_number_total)])
    now = datetime.now()
    print(str(now) + " Sending request ,", message)
    socket.send(message)

    #  Get the reply.
    message = socket.recv()
    now = datetime.now()
    print(str(now) + " Received reply ")

    # message_string = message.decode('utf8')
    # array_string = message_string.split('+')[1]
    array = np.frombuffer(message)
    return array


if __name__ == '__main__':
    # test_ping()
    array = request_rnd(10)
    print(array)