import os
from pydub import AudioSegment

import pandas as pd


def get_transcription_list_from_file(transcription_file_path: str):

    # Definition of final value
    transcriptions = []

    with open(transcription_file_path) as file:

        # Defines variables for the run to work
        current_text = ""
        line = file.readline()

        while line:

            # In the weird case of a break line, replace with space. All lines end with breaklines, those are replaced and stripped too
            line = line.replace("\n", " ").strip()

            if line != "":
                current_text += line if current_text == "" else line + "\n"
            # When the line is empty, dump whatever value is stored in the 'current_text' variable and reset the value
            else:
                if current_text != "":
                    transcriptions.append(current_text)
                    current_text = ""

            line = file.readline()

        # Process final line if not processed
        if current_text != "":
            transcriptions.append(current_text)

    return transcriptions


def process_audio_channel_folder(root_directory: str, audio_folder_name: str, channel_name:str, labels_df: pd.DataFrame, overwrite_df: bool):

    print(f"Processing channel: {channel_name}")

    # Generates the folder structure and obtains the audio files
    channel_folder = root_directory + audio_folder_name + "/" + channel_name
    files = os.listdir(channel_folder)

    # Gets the files accordingly
    audio_files = [file for file in files if file.endswith(".wav") or file.endswith(".flac")]
    transcription_file = [file for file in files if file.endswith(".txt")][0]

    # Gets the transcriptions from the text file
    transcriptions = get_transcription_list_from_file(transcription_file_path=channel_folder + "/" + transcription_file)

    # Sorts the audio files as done in the transcription process
    audio_files = sorted(audio_files, key=lambda filename: int(filename.split("part_")[1].split(".")[0]))

    # Validates the length of the files list
    if len(audio_files) == 0:
        print("There are no audios in this channel")
        return labels_df

    else:  # Process each audio
        for audio_part in range(len(audio_files)):
            file = audio_files[audio_part]
            file_path = root_directory + audio_folder_name + "/" + channel_name + "/" + file

            # Create AudioSegment instance
            audio_segment = AudioSegment.from_file(file_path)
            length = audio_segment.duration_seconds

            part_row = {
                "id_audio": audio_folder_name,
                "length": length,
                "speaker": channel_name,
                "segment_id": audio_part,
                "file_path": "{data_root}" + f"/audio_files/{audio_folder_name}/{channel_name}/{file}",
                "text": transcriptions[audio_part]
            }

            labels_df = labels_df.append(part_row, ignore_index=True)

        return labels_df


def process_audio_folder(root_directory: str, audio_folder_name: str, labels_df: pd.DataFrame, overwrite_df: bool):

    print(f"Processing folder: {audio_folder_name}")

    audio_folder = root_directory + audio_folder_name

    channel_folders = os.listdir(audio_folder)

    if len(channel_folders) == 0:
        print("This folder is empty.")
        return labels_df

    else:

        # If overwriting, filters the files
        if overwrite_df:
            labels_df = labels_df[labels_df["id_audio"] != audio_folder_name]
            print("Overwriting the audio entries in the dataset labels csv file.")

        # Process each channel only if possible according to parameters
        if overwrite_df or audio_folder_name not in labels_df["id_audio"].tolist():
            for channel_folder in channel_folders:

                labels_df = process_audio_channel_folder(
                    root_directory=root_directory,
                    audio_folder_name=audio_folder_name,
                    channel_name=channel_folder,
                    labels_df=labels_df,
                    overwrite_df=overwrite_df
                )

            return labels_df

        else:

            print("The audio already exists on the labels csv file. To overwrite, set the flag accordingly.")
            return labels_df


def prepare_dataset(root_directory: str, overwrite_df: bool = True, dest_directory: str = None):

    if not os.path.exists(root_directory):
        print("The root directory does not exist.")
        return False
    elif len(os.listdir(root_directory)) == 0:
        print("The root directory is empty")
        return False
    else:

        audio_folders = os.listdir(root_directory)

        # Obtains or generates the pandas dataframe from a csv file
        try:
            if dest_directory:
                labels_df = pd.read_csv(dest_directory + "dataset_labels.csv")
        except:
            print("First time generating labels file")
            labels_df = pd.DataFrame(columns=["id_audio", "length", "speaker", "segment_id", "file_path", "text"])

        # Process each audio folder
        for audio_folder in audio_folders:

            labels_df = process_audio_folder(
                root_directory=root_directory,
                audio_folder_name=audio_folder,
                labels_df=labels_df,
                overwrite_df=overwrite_df
            )

        # After the process is done, dumps the file
        if dest_directory:
            labels_df.to_csv(dest_directory + "dataset_labels.csv", index=False)
            return True
        else:
            labels_df.to_csv(root_directory + "dataset_labels.csv", index=False)
