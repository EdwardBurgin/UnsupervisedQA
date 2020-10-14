import spacy
import torch
import allennlp
from allennlp.models.archival import load_archive


CONSTITUENCY_MODEL = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
CONSTITUENCY_CUDA = 0
print('torch cuda available?',torch.cuda.is_available())
print(torch.version.cuda, 'cuda version')
try:
    nlp=spacy.load('en')
    print('spacy en loaded')
except Exception as e:
    print('spacy problem', e)

try:
    load_archive(CONSTITUENCY_MODEL, cuda_device=CONSTITUENCY_CUDA)
    print('allen nlp elmo model loaded correctly')
except Exception as e:
    print('ALLEN NLP CUDA PROBLEM', e)

#from allennlp.predictors.predictor import Predictor
#import allennlp_models.structured_prediction

#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
#pred = predictor.predict(
#  sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
#)
#print('ed pred',pred)
try:
    t = open('example_input.txt').read()
    if len(t) > 50:
        print('text file loaded and read correctly')
    else:
        print('asci loading error')
except Exception as e:
    print('couldnt load text file',e)
