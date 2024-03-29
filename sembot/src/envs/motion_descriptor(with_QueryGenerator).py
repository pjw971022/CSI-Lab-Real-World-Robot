import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import openai
openai.api_key = "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"
from openai import OpenAI

location = "asia-northeast3"
project_id = "gemini-api-415903"
key_path = "/home/shyuni5/file/CORL2024/Sembot/gemini-api-415903-0f8224218c2c.json"
api_key = "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"

# Video & img path는 아래와 같다고 가정.
# img, video 변수를 통해서 각 파일의 이름을 넘겨주면됨.

image_path = "/home/shyuni5/file/CORL2024/Sembot/low_level_planner/src/visualizations/obs/"
video_path = "/home/shyuni5/file/CORL2024/Sembot/low_level_planner/src/envs/"

#-----------------------------------------------------------------------------


prompt_querygenerator = '''
You are a robot manipulation assistant that make queries for video retrieval and generating context.
To carry out tasks accurately and effectively, you must learn from observing experts. This involves two steps:
Identify the types of videos to find (A): This specifies the kind of expert actions or demonstrations you should look for in videos.
Determine the details to extract from these videos (B): This outlines the specific aspects or techniques you need to understand and replicate from the video, such as movements, methods, and strategies.
Fill in A and B exactly the same as the example provided below.

For instance:
Task: Pour water from a plastic bottle into a cup held by someone.
A1: Look for videos showing people pouring water.
B: ["hand and arm movements", "The angle and speed of pouring", "The way the cup is held", "Techniques to avoid spills"]
A2: 

Task: Open the microwave door
A: Look for videos demonstrating the operation of a microwave, specifically focusing on those showing how to open and close the microwave door.
B: ["Observe the approach to the microwave", "the positioning of the hand(s) on the door", "whether a button, handle, or a combination is used to open the door", "the amount of force applied"]

Task: Open drawer.
A: Search for videos showcasing people opening various types of drawers, including kitchen, office, and bedroom drawers.
B:  ["Focus on the method of gripping the drawer handle or edge", "the direction and amount of force used to pull the drawer", "techniques to ensure smooth opening and closing"]

Task: Take usb out of computer
A: Look for instructional videos on connecting and disconnecting USB devices, with a focus on USB drives.
B:  ["Pay attention to the approach to the USB port", "the grip on the USB device", "the angle of removal", "the amount of force used to safely remove the device without damaging the port or the device", "any device-specific procedures like safely ejecting the device through the computer's operating system before physical removal."]

Task: Use the sponge to clean up the desk
A: Search for videos on cleaning surfaces, especially focusing on desks or similar furniture, using sponges or cloths.
B: ["Observe the preparation of the cleaning area", "the technique of wiping (circular motions, back and forth, etc.)", "the amount of pressure applied", "the speed of wiping"]

Complete the blanks below based on the examples above.

Task: {}
A:
'''

prompt_queryexplain ='''
Content is "{}".
Explain the content through video.
'''

#-----------------------------------------------------------------------------

def upload_blob(source_file_name, destination_blob_name, bucket_name = 'expert_video_demo'):
    storage_client = storage.Client(project='gemini-api-413603')
    bucket = storage_client.bucket(bucket_name)
    
    # 파일 업로드
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    # 파일에 공개 액세스 권한 설정
    blob.make_public()
    
    # 파일의 URI 반환
    return f"gs://{bucket_name}/{destination_blob_name}"

#-----------------------------------------------------------------------------

class MotionDescriptor_WithQuery:
    def __init__(self,) -> None:
        self.vision_config = {"max_output_tokens": 800, "temperature": 0.0, "top_p": 1, "top_k": 32}
        self.text_config = {"max_output_tokens": 512, "temperature": 0.0, "top_p": 1}
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.model = self.setup(location, project_id)

        self.gpt_assistant_prompt = 'You are a planner of a robot arm for manipulation task.'
        self.temperature=0.0
        self.max_tokens=512


    def setup(self, location, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model
    
    def gemini_querygenerator(self, user_command):
        prompt_querygenerator_with_instruction = prompt_querygenerator.format(user_command)

        contents = [prompt_querygenerator_with_instruction]
        response = self.model.generate_content(contents, generation_config=self.text_config) #, safety_settings = self.safety_settings
        return response.text

    
    def gpt4_querygenerator(self, user_command):
        prompt_querygenerator_with_instruction = prompt_querygenerator.format(user_command)
        message=[{"role": "assistant", "content": self.gpt_assistant_prompt}, {"role": "user", "content": prompt_querygenerator_with_instruction}]
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages = message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
    
descriptor = MotionDescriptor_WithQuery()

user_command = "Organize the hammer on the desk."


print("Testing gemini_querygenerator:")
print(descriptor.gemini_querygenerator(user_command))
print()
print()

print("Testing gpt4_querygenerator:")
print(descriptor.gpt4_querygenerator(user_command))





