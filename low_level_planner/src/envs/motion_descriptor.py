# import re
# import google.generativeai as genai
# genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

location = "asia-northeast3"
project_id = "gemini-api-415903"
key_path = "/home/shyuni5/file/CORL2024/Sembot/gemini-api-415903-0f8224218c2c.json"


# 아래 Link는 추후 hyperparameter로 변경 필요
# 지금은 파라미터상의 img, video 변수가 사용되지 않고 있음.

video_uri = "gs://expert_video_demo/demo.mp4"
image_uri = "gs://expert_video_demo/front_rgb.png"


#-----------------------------------------------------------------------------


prompt_descriptor = """
I am a one-armed robot. Taks is {}
Describe the motion of robot arm using the following form:

[start of description]

Speed: The robot arm should move at a speed of [NUM: 0.0]m/s.
Required Force: The robot arm should apply a force of [NUM: 0.0] Newtons.
[optional] Approach object: The robot arm should approach the object at a speed of [NUM: 0.0]m/s.
[optional] Initial Tilt(degree): The initial tilt of the robot arm should be [NUM: 0.0] degrees towards the target.
[optional] Max Tilt(degree): The maximum tilt of the robot arm should be [NUM: 0.0] degrees.
[optional] Distance Moved: The robot arm should move a distance of [NUM: 0.0] meters.
[optional] Height: The robot arm's end effector should reach a height of [NUM: 0.0] meters.
[optional] Repetitive Actions: The robot arm should perform the action [NUM: 0.0] times.
[optional] Cautions: The robot arm should {{CHOICE: [avoid obstacles, maintain a steady grip, not exceed maximum load]}}.

[end of description]

1. If you see phrases like [NUM: default_value], replace the entire phrase with a numerical value. If you see [PNUM: default_value], replace it with a positive, non-zero numerical value.
2. If you see phrases like {{CHOICE: [choice1, choice2, ...]}}, it means you should replace the entire phrase with one of the choices listed. Be sure to replace all of them. If you are not sure about the value, just use your best judgement.
3. I will tell you a behavior/skill/task that I want the robot arm to perform and you will provide the full description of the arm motion, even if you may only need to change a few lines. Always start the description with [start of description] and end it with [end of description].
4. You can assume that the robot is capable of doing anything, even for the most challenging task.
5. Do not add additional descriptions not shown above. Only use the bullet points given in the template.

"""

prompt_obs_extractor ="""
Please provide a detailed description of the photo, focusing on the objects present, their relative positions, sizes, colors, and any notable features.
Include information on the background and foreground elements, highlighting any interactions or relationships between objects.
Mention lighting conditions, textures, and materials to offer a comprehensive understanding of the scene.
Following the instruction {}, identify relevant details that could influence the task's outcome.
Adapt the description to accommodate a wide range of instructions, ensuring a thorough analysis of elements critical for task execution.
"""

prompt_demo_extractor ="""
Please provide a detailed technical analysis of the task being performed in the video, focusing specifically on the operational mechanisms and numerical data involved. The task is "{}." This analysis should include:

A step-by-step breakdown of how the task is executed, highlighting any sequential actions and the technical specifications of each step.
Precise measurements, quantities, and numerical data related to the task, such as speeds, distances, weights, and time intervals.
A detailed description of the techniques or methodologies employed in the task, emphasizing any algorithms, formulas, or calculation methods used to achieve specific outcomes.
Exclude any broader implications, potential impacts, or subjective interpretations of the task's significance. Instead, concentrate solely on providing a comprehensive, numerical, and technical description of how the task is performed, based on the content of the video.
"""
#-----------------------------------------------------------------------------

class MotionDescriptor:
    def __init__(self,) -> None:
        self.vision_config = {"max_output_tokens": 800, "temperature": 0.0, "top_p": 1, "top_k": 32}
        self.text_config = {"max_output_tokens": 512, "temperature": 0.0, "top_p": 1}
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.video = Part.from_uri(video_uri, mime_type="video/mp4")
        self.Image = Part.from_uri(image_uri, mime_type="image/png")
        self.model = self.setup(location, project_id)

    def setup(self, location, project_id):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        vertexai.init(project=project_id, location=location)
        vision_model = GenerativeModel("gemini-pro-vision")
        return vision_model
    
    def gemini_gen_u2c(self, user_command):
        prompt_descriptor_with_instruction = prompt_descriptor.format(user_command)

        contents = [prompt_descriptor_with_instruction]
        response = self.model.generate_content(contents, generation_config=self.text_config) #, safety_settings = self.safety_settings
        return response.text

    def gemini_gen_o2c(self, img, user_command):
        prompt_descriptor_with_instruction = prompt_descriptor.format(user_command)
        prompt_obs_extractor_with_instruction = prompt_obs_extractor.format(user_command)

        contents = [self.Image, prompt_obs_extractor_with_instruction ]
        response_1 = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_image = response_1.text

        contents = ["[Current image description]\n" + description_image,  prompt_descriptor_with_instruction]
        response_2 = self.model.generate_content(contents, generation_config=self.text_config) #, safety_settings = self.safety_settings
        return response_2.text
    
    def gemini_gen_d2c(self, video, img, user_command):
        prompt_descriptor_with_instruction = prompt_descriptor.format(user_command)
        prompt_obs_extractor_with_instruction = prompt_obs_extractor.format(user_command)
        prompt_demo_extractor_with_instruction = prompt_demo_extractor.format(user_command)

        contents = [self.Image, prompt_obs_extractor_with_instruction ]
        response_1 = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_image = response_1.text

        contents = [self.video, prompt_demo_extractor_with_instruction ]
        response_2 = self.model.generate_content(contents, generation_config=self.vision_config) #, safety_settings = self.safety_settings
        description_video = response_2.text

        contents = ["[Current image description]\n" + description_image + "\n", "[Goal description]\n" + description_video + "\n",  prompt_descriptor_with_instruction]
        response_3 = self.model.generate_content(contents, generation_config=self.text_config) #, safety_settings = self.safety_settings
        return response_3.text
    

# #TEST

# descriptor = MotionDescriptor()

# # Test gemini_gen_u2c
# user_command = "How can I accomplish the task of pouring liquid from a red cup into a maroon cup?"


# print("Testing gemini_gen_u2c:")
# print(descriptor.gemini_gen_u2c(user_command))

# # Test gemini_gen_o2c
# print("\nTesting gemini_gen_o2c:")
# img = "test_image.png"
# print(descriptor.gemini_gen_o2c(img, user_command))

# # Test gemini_gen_d2c
# print("\nTesting gemini_gen_d2c:")
# video = "test_video.mp4"
# print(descriptor.gemini_gen_d2c(video, img, user_command))