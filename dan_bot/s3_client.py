from os import getenv
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
