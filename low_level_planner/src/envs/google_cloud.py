from google.cloud import storage
import os

def upload_blob(source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/shyuni5/file/CORL2024/Sembot/gemini-api-415903-0f8224218c2c.json"
    storage_client = storage.Client(project='gemini-api-413603')
    bucket = storage_client.bucket(bucket_name)
    
    # 파일 업로드
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    # 파일에 공개 액세스 권한 설정
    blob.make_public()
    
    # 파일의 URI 반환
    return f"gs://{bucket_name}/{destination_blob_name}"


source_file_name = '/home/shyuni5/file/CORL2024/Sembot/low_level_planner/src/envs/Goal_0.jpg'
destination_blob_name = 'Goal_0.jpg'

public_url = upload_blob(source_file_name, destination_blob_name)
print(f'File URL: {public_url}')
