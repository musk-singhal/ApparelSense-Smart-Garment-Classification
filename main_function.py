import os
from data_gathering import capture_data
from model_prediction import predict_label
from model_training import dnn_model_training


class AutoClassification:

    def __init__(self, data_dir="img_dataset", model_dir="ml_model"):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def data_gathering(self):
        capture_data(page_range=20, dataset_path="img_dataset")

    def model_training(self):
        dnn_model_training(data_dir="img_dataset", model_save_path=r"ml_model/model_resnet18.pth")

    def prediction(self, image_url):
        predicted_gender, predicted_sleeve = predict_label(
            image_url=image_url,
            model_save_path='ml_model/model_resnet18.pth'
        )
        return predicted_gender, predicted_sleeve


if __name__ == '__main__':
    AutoClassification().data_gathering()
    AutoClassification().model_training()
    AutoClassification().prediction(image_url=r"/Users/muskan/Desktop/images/male_full_sleeve/image_1.jpg")





