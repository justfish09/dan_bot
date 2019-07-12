from os import getenv
import boto3

aws_id = getenv('DAN_BOT_AWS_ID')
aws_key = getenv('DAN_BOT_AWS_KEY')
aws_bucket = getenv('DAN_BOT_BUCKET')

s3_client = boto3.client('s3', aws_access_key_id=aws_id,
                         aws_secret_access_key=aws_key)
