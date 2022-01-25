from jiwer.transforms import SubstituteRegexes
import pandas as pd
import json
import jiwer


def labels_to_json_manifest(
        root_dir: str = ".",
        dataset_dir: str = "/benchmark_v1.5",
        file_name: str = "/dataset_labels.csv",
        export: bool = True,
        manifest_type: str = "speechbrain",
        transform_text: bool = False
):

    full_dataset_dir = root_dir + dataset_dir

    df = pd.read_csv(full_dataset_dir + file_name)

    if manifest_type == "speechbrain":

        labels_to_speechbrain_json(
            labels_df=df,
            full_dataset_dir=full_dataset_dir,
            file_name=file_name,
            export=export,
            transform_text=transform_text
        )

    elif manifest_type == "huggingface":
        labels_to_huggingface_json(
            labels_df=df,
            full_dataset_dir=full_dataset_dir,
            file_name=file_name,
            export=export,
            transform_text=transform_text
        )

    elif manifest_type == "nemo":

        labels_to_nemo_json(
            labels_df=df,
            full_dataset_dir=full_dataset_dir,
            file_name=file_name,
            export=export,
            transform_text=transform_text
        )


def labels_to_speechbrain_json(
        labels_df: pd.DataFrame,
        full_dataset_dir: str = "/benchmark_v1.5",
        file_name: str = "/dataset_labels.csv",
        export: bool = True,
        transform_text: bool = False
):

    def generate_json(labels_df, full_dataset_dir):
        dataset_json = {}
        for index, clip in labels_df.iterrows():
            # Formats the id
            id = "_".join([clip["id_audio"], clip["speaker"], str(clip["segment_id"])])

            # Adds the entry
            dataset_json[id] = {
                "file_path": clip["file_path"].replace("{data_root}", full_dataset_dir),
                "text": clip["text"] if not transform_text else process_text(clip["text"], [jiwer.ToUpperCase()]),
                "speaker": clip["speaker"],
                "length": clip["length"]
            }
        return dataset_json

    dataset_json = generate_json(labels_df, full_dataset_dir)
    dataset_json_colab = generate_json(labels_df, "/content/drive/MyDrive/Speech_Recognition/repo/fruitbat/benchmark_v1.5")

    # Exports the new json file
    if export:
        json_path = full_dataset_dir + file_name.split(".")[0] + f"_speechbrain.json"
        json_path_colab = full_dataset_dir + file_name.split(".")[0] + f"_speechbrain_colab.json"
        for path, dataset in [(json_path, dataset_json), (json_path_colab, dataset_json_colab)]:
            with open(path, mode="w") as file:
                json.dump(dataset, file)

def labels_to_huggingface_json(
        labels_df: pd.DataFrame,
        full_dataset_dir: str = "/benchmark_v1.5",
        file_name: str = "/dataset_labels.csv",
        export: bool = True,
        transform_text: bool = False
):

    def generate_json(labels_df, full_dataset_dir):
        dataset_json = {"data":[]}
        for index, clip in labels_df.iterrows():
            # Adds the entry
            dataset_json["data"] += [{
                "id": "_".join([clip["id_audio"], clip["speaker"], str(clip["segment_id"])]),
                "file_path": clip["file_path"].replace("{data_root}", full_dataset_dir),
                "text": clip["text"] if not transform_text else process_text(clip["text"], [jiwer.ToUpperCase()]),
                "speaker": clip["speaker"],
                "length": clip["length"]
            }]
        return dataset_json

    dataset_json = generate_json(labels_df, full_dataset_dir)
    dataset_json_colab = generate_json(labels_df, "/content/drive/MyDrive/Speech_Recognition/repo/fruitbat/benchmark_v1.5")

    # Exports the new json file
    if export:
        json_path = full_dataset_dir + file_name.split(".")[0] + f"_huggingface.json"
        json_path_colab = full_dataset_dir + file_name.split(".")[0] + f"_huggingface_colab.json"
        for path, dataset in [(json_path, dataset_json), (json_path_colab, dataset_json_colab)]:
            with open(path, mode="w") as file:
                json.dump(dataset, file)




def labels_to_nemo_json(
        labels_df: pd.DataFrame,
        full_dataset_dir: str = "/benchmark_v1.5",
        file_name: str = "/dataset_labels.csv",
        export: bool = True,
        transform_text: bool = False
):

    json_string = ""

    for index, clip in labels_df.iterrows():

        transcription_text = clip["text"] if not transform_text else process_text(clip["text"], [jiwer.ToLowerCase()])

        # Adds the entry
        json_line = "\"audio_filepath\": \"{}\", \"text\": \"{}\", \"duration\": {}".format(
            clip["file_path"].replace("{data_root}", full_dataset_dir).replace("\\", "/"),
            transcription_text,
            clip["length"]
        )

        # Finish format for the json line
        json_string += "{" + json_line + "}\n"

    # Exports the new json file
    if export:
        json_path = full_dataset_dir + file_name.split(".")[0] + f"_nemo.json"
        with open(json_path, mode="w") as file:
            file.write(json_string)


def process_text(line_text: str, custom_transforms = []):

    # Saves transformation pipeline
    transformations = jiwer.Compose(custom_transforms + [
        jiwer.RemoveMultipleSpaces(),
        jiwer.SubstituteRegexes({r"(\.|\,|)": ""}),  # Removes dots and commas (could add more symbols in the future)
        jiwer.SubstituteRegexes({"\n": " "}),  # Replaces the break line character with a space
        jiwer.RemoveKaldiNonWords(),  # Possibly of use with NeMo models
        jiwer.Strip()  # This should remove all leading and trailling spaces (such as those generated by the change of breakline for spaces)
    ])

    # In new transcriptions, some if not most of them end with a breakline char. We trim the transcription
    text = transformations(line_text)
    #text = text[:-1] if text.endswith(" ") else text  # Removes possible trailling 

    return text




