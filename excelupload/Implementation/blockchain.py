# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:35:01 2019

@author: Manas Ojha
"""

import datetime
import hashlib
import json
from flask import Flask, jsonify

# Part 1- Building a blockchain

class Blockchain:
    def __init__ (self):
        self.chain = []
        self.create_block(proof = 1, prev_hash = '0') #genesis block
        
    def create_block(self, proof, prev_hash):
        block = {'index' : len(self.chain) + 1,
                 'timestamp' : str(datetime.datetime.now()),
                 'proof' : proof,
                 'prev_hash' : prev_hash}
        self.chain.append(block)
        return block
    
    def get_prev_block(self):
        return self.chain[-1]
    
    def proof_of_work(self, prev_proof):
        new_proof = 1
        check_proof = False
        
        while check_proof is False:
            hash_operation = hashlib.sha256( str(new_proof**2 - prev_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True      
            else:
                new_proof += 1
                
        return new_proof
    
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys = True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
    
    def is_chain_valid(self, chain):
        prev_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['prev_hash'] != self.hash(prev_block):
                return False
            new_proof = block['proof']
            prev_proof = prev_block['proof']
            hash_operation = hashlib.sha256( str(new_proof**2 - prev_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            prev_block = block_index
            block_index += 1     
        return True

#Part 2 - Mining the blockchain
        
#Create A web App
app = Flask(__name__)
    
#creating a blockchain
blockchain = Blockchain()

#Mining a new block
@app.route('/mine_block', methods=['GET'])
def mine_block():
    prev_block = blockchain.get_prev_block()
    prev_proof = prev_block['proof']
    
    new_proof = blockchain.proof_of_work(prev_proof)
    prev_hash = blockchain.hash(prev_block)
    
    block = blockchain.create_block(new_proof, prev_hash)
    response = {'message': 'Congrats! Block Mined',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'prev_hash': block['prev_hash']}
    return jsonify(response), 201

#Getting the full blockchain
@app.route('/get_chain', methods=['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return jsonify(response), 201

#Running the app
app.run(host='0.0.0.0', port = 5000)

