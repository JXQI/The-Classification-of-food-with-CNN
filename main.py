import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--des',action="store_true",help='the code using cnn class the food')
parser.add_argument('--epoch',default=10,type=5,help='the epoch of train')
parser.add_argument('--train',default='train',type=str,help='train or test')

args=parser.parse_args()
