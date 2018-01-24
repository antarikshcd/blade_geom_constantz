''' Module containing functions to store and load pickles.
    @Author: Antariksh Dicholkar
    @Date: 8 November 2017
'''
import pickle
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)        
