from flask import Flask, render_template, request, jsonify
import pandas as pd

# import AI module
# load trained modul
# predict fr 10 values
# explainations fr 10 values
# call

#importing blockchain basics
import datetime
import hashlib
import json
import requests
from uuid import uuid4
from urllib.parse import urlparse


# Part 1 - Building a Blockchain

class Blockchain:

    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof = 1, previous_hash = '0')
        self.node = set()

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash,
                 'transactions': self.transactions}
        self.transactions = []
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof
    
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys = True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
    
    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True
    
    def add_transactions (self, transaction_keys, dataDict):
        self.transactions.append(transaction_keys)
        self.transactions.append(dataDict)
        prev_block = self.get_previous_block()
        return prev_block['index'] + 1
    
    def add_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        
    def replace_chain(self):
        network = self.node
        longest_chain = None
        max_length = len(self.chain)
        
        for node in network:
            response = request.get(f'http://{node}/get_chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length & self.is_chain_valid(chain):
                    max_length = length
                    longest_chain = chain
        if longest_chain:
            return True
        return False

# Part 2 - Mining our Blockchain

# Creating a Blockchain
blockchain = Blockchain()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

excelData = pd.DataFrame()
dataDict = ()
fileHeader = []

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        global excelData
        global dataDict
        file = request.form['upload-file']
        excelData = pd.read_excel(file)
        dataTranspose = excelData.T
        dataDict = dataTranspose.to_dict()   
        return (render_template('data.html', data=dataTranspose.to_html()))

#testing data
@app.route('/test')
def test():
    global excelData
    global dataDict
    for x in dataDict:
        print(x)
    return "All OK",200

#Parsing the urls
node_address = str(uuid4()).replace('-','')

# Mining a new block
@app.route('/mine_block', methods = ['GET'])
def mine_block():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    # blockchain.add_transactions( transaction_keys, dataDict )
    block = blockchain.create_block(proof, previous_hash)
    response = {'message': 'Congratulations, you just mined a block!',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash'],
                'transactions': block['transactions']}
    return  (render_template('mine_block.html', block=response)), 200
    #jsonify(response)

# Getting the full Blockchain
@app.route('/get_chain', methods = ['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return (render_template('get_chain.html', chains=response['chain'])), 200
    # jsonify(response)
# Checking if the Blockchain is valid
@app.route('/is_valid', methods = ['GET'])
def is_valid():
    is_valid = blockchain.is_chain_valid(blockchain.chain)
    if is_valid:
        response = {'message': 'All good. The Blockchain is valid.'}
    else:
        response = {'message': 'Houston, we have a problem. The Blockchain is not valid.'}
    return jsonify(response), 200

@app.route('/add_transaction', methods = ['GET', 'POST'])
def add_transaction():
    if request.method == 'POST':
        global dataDict
        global excelData 
        
        transaction_keys = []
        for x in dataDict:
            transaction_keys.append(x)

        if not all (key in dataDict for key in transaction_keys):
            return "Some keys are missing", 400
        # index = blockchain.add_transactions(json['sender'],json['receiver'], json['amount'])
        index = blockchain.add_transactions(transaction_keys, dataDict)
        response = {'message': f'This transaction will be added to block #{index}'}
        return (render_template('add_transaction.html', msg=response['message']))
        # jsonify(response)

@app.route('/connect_node', methods = ['POST'])
def connect_node():
    json = request.get_json()
    nodes = json.get("nodes")
    if nodes in json is None:
        return 'No Node', 400
    for node in nodes:
        blockchain.add_node(node)
    response = {'message': 'Nodes added successfully. The nodes in Xcoin are:',
                'total_nodes': list(blockchain.nodes)}
    return jsonify(response), 201
    
@app.route('/replace_chain', methods = ['GET'])
def replace_chain():
    is_chain_replaced = blockchain.replace_chain()
    if is_chain_replaced:
        response = {'message': 'Chain was replaced by the longest one',
                    'new_chain': blockchain.chain}
    else:
        response = {'message': 'All Good. Chain is the longest one',
                    'actual_chain': blockchain.chain}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, port = 5000)