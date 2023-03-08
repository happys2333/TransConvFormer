import os

self_path = os.path.dirname(__file__)


def path_parse(target_path):
    return os.path.join(self_path, target_path)


DATASET = 'ETTh1'

EPOCH = 200

LEARN = 0.0001
BATCH_SIZE = 32
SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 720

FEATURES = 'M'


PATIENCE = 10
OUTPUT_MODEL_PATH = '../output/model'

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

# result path
ECL_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ECL_ftS_sl168_ll168_pl168_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ECL_168_168_Autoformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/ECL_336_168_Reformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/ECL_S_is1_hs512_os1_sl168_pl168")
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ECL_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ECL_96_192_Autoformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/ECL_M_is321_hs512_os321_sl192_pl192")
    }

}

WTH_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_WTH_ftS_sl168_ll168_pl168_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/WTH_168_168_Autoformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/WTH_336_168_Reformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/WTH_S_is1_hs512_os1_sl168_pl168")
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_WTH_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/WTH_96_192_Autoformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/WTH_M_is12_hs512_os12_sl192_pl192")
    }

}

ETTH1_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ETTh1_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ETTh1_168_168_Autoformer_ETTh1_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/ETTh1_336_168_Reformer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/ETT_S_is1_hs512_os1_sl168_pl168")
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ETTh1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ETTh1_96_192_Autoformer_ETTh1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "LSTM": path_parse("../result/output/LSTM/ETT_M_is7_hs512_os7_sl192_pl192")
    }

}
