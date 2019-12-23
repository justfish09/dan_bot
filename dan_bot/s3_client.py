from os import getenv
from pathlib import Path
import boto3


class S3Client(object):
    def __init__(self):
        self.aws_id = getenv('DAN_BOT_AWS_ID')
        self.aws_key = getenv('DAN_BOT_AWS_KEY')
        self.aws_bucket = getenv('DAN_BOT_BUCKET')

    def client(self):
        return boto3.client('s3', aws_access_key_id=self.aws_id,
                            aws_secret_access_key=self.aws_key)

    def resource(self):
        return boto3.resource('s3', aws_access_key_id=self.aws_id,
                              aws_secret_access_key=self.aws_key)

def save_data_locally():
    mod_path = Path(__file__).parent
    path_to_file = (mod_path / '../input_data')
    s3_client = S3Client()
    bucket = s3_client.resource().Bucket(s3_client.aws_bucket)
    bucket.download_file('input_data/dan_bot.zip', str(path_to_file / 'dan_bot.zip'))
    bucket.download_file('input_data/keras_model.h5', str(path_to_file / 'keras_model.h5'))
    bucket.download_file('input_data/tf_lite_model.tflite', str(path_to_file / 'tf_lite_model.tflite'))

if __name__ == '__main__':
    save_data_locally()
