import re
import google.generativeai as genai
genai.configure(api_key='AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I')
prompt_descriptor = """
Objective Description:
I am a one-armed robot. How can I accomplish the task of pouring liquid from a red cup into a maroon cup?

Required Action Summary:
Please provide only the essential motions, without unnecessary explanations.

Detailed Motion Information:

Action Name:
[optional] Approach object:
[optional] Initial Tilt(degree):
[optional] Max Tilt(degree):
[optional] Required Force:
[optional] Distance Moved:
[optional] Height:
[optional] Speed:
[optional] Repetitive Actions:
[optional] Cautions:
Focus solely on the motion information, including detailed information such as angle, force, and distance.
"""
prompt_obs_extractor ="""
"""
prompt_demo_extractor ="""
"""
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

    def gemini_gen_u2c(self, user_command): # User Command
        model = genai.GenerativeModel('gemini-pro')
        upstream_message = [prompt_obs_extractor + '\nTask: ' + user_command]
        response = model.generate_content(
            contents=upstream_message,
            generation_config=self.text_config, 
            safety_settings = self.safety_settings
        )
        parts = response.parts
        motion_guideline = ''
        for part in parts:
            motion_guideline += part.text
        return motion_guideline
    
    def gemini_gen_o2c(self, img, user_command): # Vision observation / User Command
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        upstream_message = [prompt_obs_extractor + '\nTask: ' + user_command, img]
        response0 = vision_model.generate_content(
            contents=upstream_message,
            generation_config=self.vision_config, 
            safety_settings = self.safety_settings
        )
        parts = response0.parts
        vis_obs_context = ''
        for part in parts:
            vis_obs_context += part.text
        
        upstream_message = [prompt_obs_extractor + '\nTask: ' + user_command + '\n' + vis_obs_context] # @
        response1 = vision_model.generate_content(
            contents=upstream_message,
            generation_config=self.vision_config, 
            safety_settings = self.safety_settings
        )
        parts = response1.parts
        motion_guideline = ''
        for part in parts:
            motion_guideline += part.text
        return motion_guideline
    
    def gemini_gen_d2c(self, video, img, user_command): # Expert Demo / Vision observation / User Command # @ vertaxAI
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        upstream_message = [prompt_obs_extractor + '\nTask: ' + user_command, img]
        response0 = vision_model.generate_content(
            contents=upstream_message,
            generation_config=self.vision_config, 
            safety_settings = self.safety_settings
        )
        parts = response0.parts
        vis_obs_context = ''
        for part in parts:
            vis_obs_context += part.text

        upstream_message = [prompt_demo_extractor + '\nTask: ' + user_command, video]
        response = vision_model.generate_content(
            contents=upstream_message,
            generation_config=self.vision_config,
            safety_settings = self.safety_settings
        )
        parts = response.parts
        demo_context = ''
        for part in parts:
            demo_context += part.text

        model = genai.GenerativeModel('gemini-pro')
        upstream_message = [prompt_obs_extractor + '\nTask: ' + user_command, demo_context, vis_obs_context]
        response = model.generate_content(
            contents=upstream_message,
            generation_config=self.text_config, 
            safety_settings = self.safety_settings
        )
        parts = response.parts
        motion_guideline = ''
        for part in parts:
            motion_guideline += part.text
        return motion_guideline

