
import glob
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from IPython.display import display
import PIL
# import fitz
import numpy as np
import pandas as pd
import requests
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part    
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Video as vision_model_Video

from vertexai.vision_models import MultiModalEmbeddingModel
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)
from sembot.src.VideoRAG.utils.intro_multimodal_rag_utils import (
    get_pdf_doc_object,
    get_chunk_text_metadata,
    get_image_for_gemini,
    get_gemini_response,
    get_image_metadata_df,
    get_text_metadata_df,
    get_text_embedding_from_text_embedding_model,
    get_image_embedding_from_multimodal_embedding_model,
    get_cosine_score,
    get_user_query_text_embeddings,
    )
from google.cloud import storage
import vertexai
location = "asia-northeast3"
project_id = "gemini-api-415903"

def upload_blob(source_file_name, destination_blob_name, bucket_name='human_video_demo'):
    storage_client = storage.Client(project='gemini-api-413603')
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    blob.make_public()    
    return f"gs://{bucket_name}/{destination_blob_name}"



def get_video_for_gemini(
    file_name: str,
) -> Tuple[vision_model_Video, str]:
    """
    Extracts a video from a PDF document, saves it to a specified directory,
    and loads it as a vision_model_Video object.

    Parameters:
    - video_save_dir (str): The directory where the video will be saved.
    - file_name (str): The base name for the video file.

    Returns:
    - Tuple[vision_model_Video, str]: A tuple containing the vision_model_Video object and the video filename.
    """

    # Load the saved video as a vision_model_Video object
    video_uri = upload_blob(file_name, file_name.split("/")[-1])
    video_for_gemini = Part.from_uri(video_uri, mime_type="video/mp4")

    return video_for_gemini, file_name
    
def get_video_embedding_from_multimodal_embedding_model(
    video_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts a video embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        video_uri (str): The URI (Uniform Resource Identifier) of the video to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the video embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    # video = Video.load_from_file(video_uri)
    video = vision_model_Video.load_from_file(video_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        video=video, contextual_text=text, dimension=embedding_size
    )  # 128, 256, 512, 1408
    video_embedding = embeddings.video_embedding

    if return_array:
        video_embedding = np.fromiter(video_embedding, dtype=float)

    return video_embedding


def get_user_query_video_embeddings(
    video_query_path: str, embedding_size: int
) -> np.ndarray:
    """
    Extracts video embeddings for the user query video using a multimodal embedding model.

    Args:
        video_query_path: The path to the user query video.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query video embedding.
    """

    return get_video_embedding_from_multimodal_embedding_model(
        video_uri=video_query_path, embedding_size=embedding_size
    )


def get_video_metadata_df(
    filename: str, video_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a video metadata dictionary as input,
    iterates over the video metadata dictionary and extracts the video path,
    video description, and video embeddings for each video, creates a Pandas
    DataFrame with the extracted data, and returns it.

    Args:
        filename: The filename of the document.
        video_metadata: A dictionary containing the video metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted video path, video description, and video embeddings for each video.
    """

    final_data_video: List[Dict] = []
    for key, video_values in video_metadata.items():
        # for _, video_values in values.items():
        data: Dict = {}
        data["file_name"] = filename
        data["video_no"] = int(video_values["video_no"])
        data["video_path"] = video_values["video_path"]
        data["video_desc"] = video_values["video_desc"]
        data["video_embedding_from_video_only"] = video_values[
            "video_embedding_from_video_only"
        ]
        data["text_embedding_from_video_description"] = video_values[
            "text_embedding_from_video_description"
        ]
        final_data_video.append(data)

    return_df = pd.DataFrame(final_data_video).dropna()
    return_df = return_df.reset_index(drop=True)
    return return_df

def get_similar_video_from_query(
    video_metadata_df: pd.DataFrame,
    query: str = "",
    video_query_path: str = "",
    column_name: str = "",
    video_emb: bool = True,
    top_n: int = 3,
    embedding_size: int = 512,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar videos from a metadata DataFrame based on a text query or a video query.

    Args:
        text_metadata_df: A Pandas DataFrame containing text metadata associated with the videos.
        video_metadata_df: A Pandas DataFrame containing video metadata (paths, descriptions, etc.).
        query: The text query used for finding similar videos (if video_emb is False).
        video_query_path: The path to the video used for finding similar videos (if video_emb is True).
        column_name: The column name in the video_metadata_df containing the video embeddings or captions.
        video_emb: Whether to use video embeddings (True) or text captions (False) for comparisons.
        top_n: The number of most similar videos to return.
        embedding_size: The dimensionality of the video embeddings (only used if video_emb is True).

    Returns:
        A dictionary containing information about the top N most similar videos, including cosine scores, video objects, paths, page numbers, text excerpts, and descriptions.
    """
    # Check if video embedding is used
    if video_emb:
        # Calculate cosine similarity between query video and metadata videos
        user_query_video_embedding = get_user_query_video_embeddings(
            video_query_path, embedding_size
        )
        cosine_scores = video_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_video_embedding),
            axis=1,
        )
    else:
        # Calculate cosine similarity between query text and metadata video captions
        user_query_text_embedding = get_user_query_text_embeddings(query)
        cosine_scores = video_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_text_embedding),
            axis=1,
        )

    # Remove same video comparison score when user video is matched exactly with metadata video
    cosine_scores = cosine_scores[cosine_scores < 1.0]

    # Get top N cosine scores and their indices
    top_n_cosine_scores = cosine_scores.nlargest(top_n).index.tolist()
    top_n_cosine_values = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched videos and their information
    final_videos: Dict[int, Dict[str, Any]] = {}

    for matched_videonum, indexvalue in enumerate(top_n_cosine_scores):
        # Create a sub-dictionary for each matched video
        final_videos[matched_videonum] = {}

        # Store cosine score
        final_videos[matched_videonum]["cosine_score"] = top_n_cosine_values[
            matched_videonum
        ]

        # Add file name
        final_videos[matched_videonum]["file_name"] = video_metadata_df.iloc[indexvalue][
            "file_name"
        ]

        # Store video path
        final_videos[matched_videonum]["video_path"] = video_metadata_df.iloc[indexvalue][
            "video_path"
        ]

        # Store video description
        final_videos[matched_videonum]["video_description"] = video_metadata_df.iloc[
            indexvalue
        ]["video_desc"]

    return final_videos

def get_CharadesEgo_document_metadata(
    generative_multimodal_model,
    video_folder_path: str,
    video_description_prompt: str,
    embedding_size: int = 128,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    add_sleep_after_video: bool = False,
    sleep_time_after_video: int = 2,
    use_video_description: bool = True,
    anntotation_path: str = None,
    metadata_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a PDF path, an video save directory, an video description prompt, an embedding size, and a text embedding text limit as input.

    Args:
        video_save_dir: The directory where extracted videos should be saved.
        video_description_prompt: A prompt to guide Gemini for generating video descriptions.
        embedding_size: The dimensionality of the embedding vectors.
        text_emb_text_limit: The maximum number of tokens for text embedding.

    Returns:
        A tuple containing two DataFrames:
            * One DataFrame containing the extracted text metadata for each page of the PDF, including the page text, chunked text dictionaries, and chunk embedding dictionaries.
            * Another DataFrame containing the extracted video metadata for each video in the PDF, including the video path, video description, video embeddings (with and without context), and video description text embedding.
    """
    
    annotation_df = pd.read_csv(anntotation_path)
    video_metadata_df_final = pd.DataFrame()

    for video_no, video_file in enumerate(glob.glob(video_folder_path + "/*.mp4")): # @ TODO
        video_metadata: Dict[Union[int, str], Dict] = {}

        video_number = int(video_no + 1)
        video_metadata[video_number] = {}

        video_for_gemini, video_name = get_video_for_gemini(
            video_file
        )
        video_embedding = get_video_embedding_from_multimodal_embedding_model(
            video_uri=video_name,
            embedding_size=embedding_size,
        )

        if  use_video_description:  
            response = get_gemini_response(
                generative_multimodal_model,
                model_input=[video_description_prompt, video_for_gemini],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )

            video_description_text_embedding = (
                get_text_embedding_from_text_embedding_model(text=response)
            )
        else:
            video_description_text_embedding = None
            response = None
        
        video_annotation = annotation_df['id'==video_file.split('.')[0]].to_dict() # @
        
        video_metadata[video_number] = {
            "video_num": video_number,
            "video_path": video_name,
            "video_desc": response,
            'video_annotation': video_annotation,
            # "mm_embedding_from_text_desc_and_video": video_embedding_with_description,
            "mm_embedding_from_video_only": video_embedding,
            "text_embedding_from_video_description": video_description_text_embedding,
        }

        # Add sleep to reduce issues with Quota error on API
        if add_sleep_after_video:
            time.sleep(sleep_time_after_video)
            print(
                "Sleeping for ",
                sleep_time_after_video,
                """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
            )

        video_metadata_df = get_video_metadata_df(video_file, video_metadata)

        # text_metadata_df_final = pd.concat(
        #     [text_metadata_df_final, text_metadata_df], axis=0
        # )
        video_metadata_df_final = pd.concat(
            [
                video_metadata_df_final,
                video_metadata_df.drop_duplicates(subset=["video_desc"]),
            ],
            axis=0,
        )

    video_metadata_df_final = video_metadata_df_final.reset_index(drop=True)

    video_metadata_df_final.to_csv(metadata_path, index=False)
    return video_metadata_df_final





def get_EK10_document_metadata(
    generative_multimodal_model,
    video_folder_path: str,
    video_description_prompt: str,
    embedding_size: int = 128,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    add_sleep_after_video: bool = False,
    sleep_time_after_video: int = 2,
    use_video_description: bool = True,
    anntotation_path: str = None,
    metadata_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a PDF path, an video save directory, an video description prompt, an embedding size, and a text embedding text limit as input.

    Args:
        video_save_dir: The directory where extracted videos should be saved.
        video_description_prompt: A prompt to guide Gemini for generating video descriptions.
        embedding_size: The dimensionality of the embedding vectors.
        text_emb_text_limit: The maximum number of tokens for text embedding.

    Returns:
        A tuple containing two DataFrames:
            * One DataFrame containing the extracted text metadata for each page of the PDF, including the page text, chunked text dictionaries, and chunk embedding dictionaries.
            * Another DataFrame containing the extracted video metadata for each video in the PDF, including the video path, video description, video embeddings (with and without context), and video description text embedding.
    """
    if anntotation_path is not None:
        annotation_df = pd.read_csv(anntotation_path)
    video_metadata_df_final = pd.DataFrame()
    video_no = 0
    for video_folder in glob.glob(video_folder_path):
        video_no += 1
        for video_file in glob.glob(video_folder + "/*.mp4"):
            video_metadata: Dict[Union[int, str], Dict] = {}

            video_number = video_no
            video_metadata[video_number] = {}

            video_for_gemini, video_name = get_video_for_gemini(
                video_file
            )
            video_embedding = get_video_embedding_from_multimodal_embedding_model(
                video_uri=video_name,
                embedding_size=embedding_size,
            )

            if  use_video_description:  
                response = get_gemini_response(
                    generative_multimodal_model,
                    model_input=[video_description_prompt, video_for_gemini],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
                )

                video_description_text_embedding = (
                    get_text_embedding_from_text_embedding_model(text=response)
                )
            else:
                video_description_text_embedding = None
                response = None
            
            if anntotation_path is not None:
                video_annotation = annotation_df['id'==video_file.split('.')[0]].to_dict() # @
            else:
                video_annotation = None
            video_metadata[video_number] = {
                "video_num": video_number,
                "video_path": video_name,
                "video_desc": response,
                'video_annotation': video_annotation,
                # "mm_embedding_from_text_desc_and_video": video_embedding_with_description,
                "mm_embedding_from_video_only": video_embedding,
                "text_embedding_from_video_description": video_description_text_embedding,
            }

            # Add sleep to reduce issues with Quota error on API
            if add_sleep_after_video:
                time.sleep(sleep_time_after_video)
                print(
                    "Sleeping for ",
                    sleep_time_after_video,
                    """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
                )

            video_metadata_df = get_video_metadata_df(video_file, video_metadata)

            # text_metadata_df_final = pd.concat(
            #     [text_metadata_df_final, text_metadata_df], axis=0
            # )
            video_metadata_df_final = pd.concat(
                [
                    video_metadata_df_final,
                    video_metadata_df.drop_duplicates(subset=["video_desc"]),
                ],
                axis=0,
            )

    video_metadata_df_final = video_metadata_df_final.reset_index(drop=True)

    video_metadata_df_final.to_csv(metadata_path, index=False)
    return video_metadata_df_final