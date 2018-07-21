import torch as ch
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TSNE(object):

    def __init__(num_components=None, perplexity=30.0):
        self.n_components = n_components
        self.perplexity = perplexity
        
