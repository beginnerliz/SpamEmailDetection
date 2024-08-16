# Creator Cui Liz
# Time 16/08/2024 03:45


from Model.Trans_Classifier import TransformerClassifier
from Model.Res_Classifier import ResClassifier
from Model.RNN_Classifier import RNNClassifier

from train import train
from eval import eval
from Dataset.EmailDataProcess import SpamIsHamDataset, createDatasetCache
import yaml
from datetime import datetime

if __name__ == "__main__":

    # with open("Dataset/DatasetPaths.yaml", "r") as in_file:
    #     train_paths = yaml.safe_load(in_file)["train_paths"]

    for LLM_chunks in [0, 1, 2, 3, 4, 5]:
        with open("Eval_Results.txt", "a") as out_file:
            time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
            # out_file.write(f"{time_now} : Transformer with {LLM_chunks} LLM chunks\n")

            out_file.write(f"{time_now} : Transformer, RNN, Res with {LLM_chunks} LLM chunks\n")

        for model_class in [RNNClassifier, ResClassifier, TransformerClassifier]:
            trained_model, best_report = train(model_class, LLM_chunks / 5)

            with open("Eval_Results.txt", "a") as out_file:
                out_file.write("\t" + best_report + "\n")

                del (trained_model)