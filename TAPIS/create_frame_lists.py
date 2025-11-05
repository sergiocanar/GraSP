import os
import pandas as pd
import json
from tqdm import tqdm


def load_json(json_path: str) -> dict:
    """
    Load a JSON file and return its content.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Content of the JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def main(
    frames_path: str,
    annotations_path: str,
    output_paht: str,
    frame_number_type: str = "frame_name",
) -> None:
    """
    Create a list of frame files in the specified directory.

    Args:
        frames_path (str): Path to the directory containing frame files.
        annotations_path (str): Path to the directory containing annotations.
        output_path (str): Path to the directory where the output CSV file will be saved.
        frame_number_type (str): Type of frame number to use. Can be either "continuous" or "frame_name".
    """
    assert frame_number_type in [
        "continuous",
        "frame_name",
    ], "frame_number_type must be either 'continuous' or 'frame_name'"

    splits = ["train", "test"]

    for split in splits:
        annotations = load_json(os.path.join(annotations_path, f"{split}.json"))
        video_list = list(
            {image["file_name"].split(os.sep)[0] for image in annotations["images"]}
        )
        video_list = sorted(video_list)

        video_name_list = []
        video_number_list = []
        frame_number_list = []
        frame_name_list = []

        for video in tqdm(video_list):
            # Get the list of frames for the current video
            frames = os.listdir(os.path.join(frames_path, video))
            frames = sorted(frames)

            for frame_number, frame in enumerate(frames):
                # Create a DataFrame for the current frame
                video_number = int(video.split("_")[-1])
                frame_name = os.path.join(video, frame)
                if frame_number_type == "frame_name":
                    frame_number = int(frame.split(".")[0])

                video_name_list.append(video)
                video_number_list.append(video_number)
                frame_number_list.append(frame_number)
                frame_name_list.append(frame_name)

        # Append the current frame DataFrame to the main DataFrame
        df = pd.DataFrame(
            {
                "VIDEO_NAME": video_name_list,
                "VIDEO_NUMBER": video_number_list,
                "FRAME_NUMBER": frame_number_list,
                "FRAME_NAME": frame_name_list,
            }
        )

        # Save the DataFrame to a CSV file
        output_csv_path = os.path.join(output_path, f"{split}.csv")
        # remove column names from the first row
        df.to_csv(output_csv_path, index=False, header=False, sep=" ")
        print(f"Frame list for {split} saved to {output_csv_path}")


if __name__ == "__main__":
    frames_path = os.path.join("..", "frames")
    annotations_path = os.path.join("..", "annotations")
    output_path = os.path.join("..", "frame_lists")
    main(frames_path, annotations_path, output_path)