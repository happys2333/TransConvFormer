import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import pandas as pd

from util.dataset import *
from util.param import *


def plot_comparison(datasets: Union[list, tuple], models: Union[list, tuple], pred_len: int, features="S", save=False,
                    fig_size=(15, 5), pred_idx=168):
    """
    pred_idx means it will predict from [-pred_idx:-pred_idx+pred_len], which is different from
    """
    plt.figure(figsize=fig_size)
    for dataset in datasets:
        ground_truth = None
        for model in models:
            gt, pre = plot_pred(pred_idx, pred_len, dataset, model, features=features, show=False)
            assert gt is not None, pre is not None
            if ground_truth is None:
                ground_truth = gt
                plt.plot(gt, label="Ground Truth")
            plt.plot(pre, label="%s" % model)

        feature_type = "Multivariate" if features == "M" else "Univariate"
        title_name = "%s prediction with length %d on %s" % (feature_type, pred_len, dataset)
        plt.title(title_name)
        plt.legend()
        if save:
            plt.savefig(title_name)

        plt.show()


def plot_pred(pred_idx=168, pred_len=24, dataset="ETTh1", model="informer", features="S", show=True, save=False):
    """
    Shows comparison between prediction of informer and ground truth of ECL's last sequence
    Firstly use ckpt of informer to predict a sequence, then use the prediction by setting 'ECL_RESULT_PATH'
    :return:
    """
    if dataset == "ETTh1":
        data = ETT().df_h1
        result_path = ETTH1_PRED[features][model]
    elif dataset == "ECL":
        data = ECL().df
        result_path = ECL_PRED[features][model]
    elif dataset == "WTH":
        data = WTH().df
        result_path = WTH_PRED[features][model]
    else:
        print("Dataset not found")
        return None, None
    if result_path is None:
        return None, None

    end_point = (-pred_idx + pred_len) if (-pred_idx + pred_len) < 0 else None
    ground_truth = np.array(data.iloc[-pred_idx:end_point, -1])
    pred = np.load(os.path.join(result_path, "real_prediction.npy"))
    if features == "M":
        target_pred = pred.squeeze()[:, -1]
    else:
        target_pred = pred.squeeze()[:]

    if show or save:
        plt.figure(figsize=(15, 5))
        plt.plot(target_pred, label="Pred")
        plt.plot(ground_truth, label="GT")
        title = "%s prediction of %s on %s dataset with %d length" % (features, model, dataset, pred_len)
        plt.title(title)
        plt.xlabel("Time (h)")
        plt.ylabel("Value")
        plt.legend()
        if save:
            plt.savefig(title+" (pre_idx:%d)" % pred_idx)
        if show:
            plt.show()

    return ground_truth, target_pred


def draw_result_bars():
    colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
              "#86BCB6", "#E15759", "#E19D9A"]
    total_width = 0.6

    def draw_subplot(model_names: list, dataset_names: list, origin_data, start_pos: int, metric_name, pre_len=192,
                     feature_type="M"):
        plot_num = len(origin_data)
        pos = start_pos
        end_pos = pos + plot_num - 1
        for p_n in range(plot_num):
            model_name = model_names[p_n]
            n = len(model_name)  # n is the number of model
            width = total_width / n  # 每种类型的柱状图宽度
            f_type_name = "multivariate" if feature_type == "M" else "univariate"

            # 重新设置x轴的坐标
            # range_x is the number of datasets
            dataset_name = dataset_names[p_n]
            range_x = len(dataset_name)
            x = np.arange(range_x)
            x = x - (total_width - width) / 2

            # 画柱状图
            plt.subplot(pos)
            for i in range(n):
                plt.bar(x + i * width, origin_data[p_n][i], width=width, label=model_name[i], color=colors[i])
            plt.xticks(np.arange(range_x), dataset_name)
            # 显示图例
            if pos == start_pos:
                plt.title("Models comparison with %s prediction of %d length" % (f_type_name, pre_len))
            plt.legend()
            if pos == end_pos:
                plt.xlabel("Datasets", fontname="Times New Roman")
            plt.ylabel("%s" % metric_name[p_n], fontname="Times New Roman")
            pos += 1

        plt.show()

    dataset = [["ECL", "ETTh1", "WTH"], ["ECL", "ETTh1", "WTH"]]

    # M
    f_t1 = "M"
    models1 = [["Autoformer", "Informer", "LSTM"], ["Autoformer", "Informer", "LSTM"]]
    pred_len1 = 192
    metric1 = ["MSE", "MAE"]
    pos1 = 211
    # dataset[i] means ith model's metric on len(dataset[i]) datasets
    data1 = [[[0.193802, 0.535738, 0.587715],
              [0.284998, 1.130280, 0.542752],
              [0.394332, 1.266324, 0.745868]],
             [[0.306151, 0.487452, 0.559554],
              [0.374030, 0.875960, 0.525745],
              [0.451243, 0.840875, 0.622514]]
             ]
    draw_subplot(model_names=models1, dataset_names=dataset, origin_data=data1, start_pos=pos1, metric_name=metric1,
                 feature_type=f_t1, pre_len=pred_len1)

    # S
    f_t2 = "S"
    models2 = [["Autoformer", "Informer", "LSTM"],
               ["Autoformer", "Informer", "LSTM"]]
    pred_len2 = 168
    metric2 = ["MSE", "MAE"]
    pos2 = 211
    # dataset[i] means ith model's metric on len(dataset[i]) datasets
    data2 = [[[0.384328, 0.142934, 0.243901],
              [0.336435, 0.235669, 0.250150],
              # [0.394037, 0.430356, 0.202001],
              [0.319112, 0.283552, 0.334202]],
             [[0.466045, 0.292151, 0.367090],
              [0.409530, 0.419669, 0.376537],
              # [0.462601, 0.585273, 0.336203],
              [0.409603, 0.424886, 0.435629]]
             ]
    draw_subplot(model_names=models2, dataset_names=dataset, origin_data=data2, start_pos=pos2, metric_name=metric2,
                 feature_type=f_t2, pre_len=pred_len2)


if __name__ == "__main__":
    plot_pred(1368, 168, "ECL", "LSTM", features="S", save=True)
    # plot_comparison(datasets=["ECL"], models=["informer", "autoformer", "LSTM"], pred_idx=500, pred_len=192,
    #                 features="M", save=True)
    # draw_result_bars()
