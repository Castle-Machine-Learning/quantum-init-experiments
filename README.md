# Quantum-Randomness-Service

Code for a quantum randomness service. 
To request random numbers or set up a server, clone this repo:
``` bash
   $ git clone https://gitlab.scai.fraunhofer.de/moritz.wolter/fzmlq-quantum-randomness-service.git
```


# Client:
If a test server is running on the local host,
random numbers can be requested from within python using:
``` python
   >>> from src import FCMLQ
   >>> FCMLQ.request_rnd(10)
```
To query a remote host say `lusin` at port `5555` use:
``` python
   >>> from src import FCMLQ
   >>> FCMLQ.request_rnd(10, address="tcp://lusin:5555")

```

# Server:

### Setting up IBMQ
Head over to https://quantum-computing.ibm.com/ and set up an account.
An account token is required to run this code. Take a look at 
https://quantum-computing.ibm.com/docs/manage/account/ to learn
where to find your access key.
When you have the token open our python interpreter and run
``` python
 >>> from qiskit import IBMQ
 >>> IBMQ.save_account('insert your IBMQ access-token here.')
```
to set up your configuration.
Once the configuration is set up, to start a test server, run:
``` bash
   $ python src/server/server.py --port 5555 --backend simulator
```
See `python src/server/server.py -h` for more options.





