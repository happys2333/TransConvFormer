import argparse
import os


self_path = os.path.dirname(__file__)


def path_parse(target_path):
    return os.path.join(self_path, target_path)




EPOCH = 200

LEARN = 0.0001
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96, help='sequence length')
parser.add_argument('--factor', type=int, default=5, help='factor')
parser.add_argument('--d_model', type=int, default=512, help='d_model')
parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
parser.add_argument('--enc_layers', type=int, default=3, help='encoder layers number')
parser.add_argument('--label_len', type=int, default=48, help='label length')
parser.add_argument('--pred_len', type=int, default=120, help='prediction length')
parser.add_argument('--features', type=str, default='M', help='features')
parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset')
args = parser.parse_args()

SEQ_LEN = args.seq_len
LABEL_LEN = args.label_len
PRED_LEN = args.pred_len
ENC_LAYER = args.enc_layers
DATASET = args.dataset
FACTORS = args.factor
D_MODEL = args.d_model
N_HEADS = args.n_heads

FEATURES = args.features


PATIENCE = 10
OUTPUT_MODEL_PATH = './output/model'

data_parser = {
    'ETTh1': {'dataset': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'dataset': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'dataset': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'dataset': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'dataset': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'dataset': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'KDD': {'dataset': 'wtbdata_245days.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
}
ENCODER_IN, DECODER_IN, OUT_SIZE = data_parser[DATASET][FEATURES]
# dataset set path

ETT_PATH_DIR = path_parse("../dataset/ETT-small")
ECL_PATH = path_parse("../dataset/ECL.csv")
WTH_PATH = path_parse("../dataset/WTH.csv")
KDD_PATH = path_parse("../dataset/wtbdata_245days.csv")

