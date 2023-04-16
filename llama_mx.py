# pip install matrix-commander
# pip install pyllamacpp

import asyncio
import json
import os
from pyllamacpp.model import Model
import nest_asyncio
from subprocess import Popen, PIPE
from time import time, sleep
nest_asyncio.apply()


class MatrixLLMBot():
    
    def __init__(self, **kwargs):
        '''Constructor.'''
        
        self.credentials_path = kwargs.get('credentials_path', '~/')
        self.model_store = kwargs.get('model_store', '~/')
        self.sleep_duration = 1
        self.prompter = LLMPrompter(model_store=self.model_store)
        
        # necessary for credentials
        os.chdir(self.credentials_path)
        
    async def receive(self):
        '''
        Receiving messages from the Matrix server.
        '''
        
        messages, room_ids = [], []
        
        process = Popen(['matrix-commander', '-l', 'ONCE', '--listen-self', '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if stdout != b'':
            msg = stdout.decode()
            msg = json.loads(msg)
            room_id = msg['source']['room_id']
            msg = msg['source']['content']['body']
            messages.append(msg)
            room_ids.append(room_id)
            # print('debug received:', msg)

        sleep(self.sleep_duration)

        return messages, room_ids
    
    def models_list(self):
        _str = ''
        for i, r in enumerate(self.prompter.response_strings):
            if i == 0:
                _str += r
            if i > 0:
                _str += '\n' + r
            
        return _str
    
    async def send(self, output, room_id):
        '''
        Sending response back to the Matrix server.
        '''
        
        # to avoid " conflicts
        output = output.replace('"', '\"')

        process = Popen(['matrix-commander', '-m', output, '--room', room_id], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

    def start(self):
        while True:
            messages, room_ids = asyncio.run(self.receive())

            output = None
            if len(messages) > 0:
                for iMessage, message in enumerate(messages):
                    # print('debug, message is:', message)
                    
                    if message == '!models':
                        output = self.models_list()
                    else:
                        try:
                            output = asyncio.run(self.prompter.generate(prompt=message))
                        except Exception as e:
                            print(e)
                    if output != None:
                        asyncio.run(self.send(output, room_ids[iMessage]))


class LLMPrompter():
    
    def __init__(self, model_store):
        '''
        Constructor.
        
        models currently from:
        https://huggingface.co/eachadea
        https://huggingface.co/TheBloke
        '''
        
        self.model_store = model_store
        self.response_strings = [
            '!toolpaca-13b',
#             '!vicuna-7b-1.0-uncensored',
#             '!vicuna-7b-1.1',
            '!gpt4all-lora',
            '!toolpaca-13b',
            ]

    def selectmodel_fromprefix(self, prefix):
        # better search from response_strings?

        model_filename = None
        model_path = None

        if prefix == '!toolpaca-13b':
            model_filename = 'ggml-toolpaca-13b-4bit.bin'
        if prefix == '!vicuna-7b-1.0-uncensored':
            model_filename = 'ggml-vicuna-7b-1.0-uncensored-q4_0.bin'
        if prefix == '!vicuna-7b-1.1':
            model_filename = 'ggml-vicuna-7b-1.1-q4_0.bin'
            # model_filename = 'ggml-vicuna-7b-1.1-q4_1.bin'
        if prefix == '!gpt4all-lora':
            model_filename = 'gpt4all-lora-quantized-ggml.bin'
        if prefix == '!toolpaca-13b':
            model_filename = 'ggml-toolpaca-13b-4bit.bin'
            
        if model_filename != None:
            model_path = os.path.join(self.model_store, model_filename)
        
        return model_path
        
    async def generate(self,
                       prompt,
                       init_prompt=''
                      ):
        '''Generating response.'''

        if prompt.split(' ')[0] in self.response_strings:
            model_path = self.selectmodel_fromprefix(prompt.split(' ')[0])
            model = Model(ggml_model=model_path,
                          n_ctx=512,
                          # n_threads=8
                         )
            output = model.generate(init_prompt+prompt, n_predict=55)
            return output
        
        else:
            return None
