import os
import sys
import subprocess
import shutil
import pandas as pd
import json
from modules.dataset_preparation.dataset_preparation import prepare_dataset
from modules.manifest_generator.manifest_generator import generateTrainValPartitions

path = os.getcwd()
p = None


def printResponse(p):
    for line in p.stdout.readlines():
        print(line)

def checkPushError(p):
    for line in p.stdout.readlines():
        if "failed to push" in str(line):
            print("Data upload failed, retrying...")
            return True
    print("No push errors encountered")
    return False

def getGitAddCommand(p):
    for line in p.stdout.readlines():
        if "git add" in str(line):
            result = str(line)[4:]
            result = result[:-3]
            result = result.replace("\\r","")
            result = result.replace("\\n","")
            return result
    return False

def getCommitID(p, file, version):
    reference = ""
    if version == "latest":
        reference = file
    else:
        reference = "{0} {1}".format(file, version)

    for line in p.stdout.readlines():
        if reference in str(line):
            return str(line)[2:10]
    return 0

def handleCheckout(commitID, file):
    try:
        shutil.rmtree(path + '\{0}'.format(file))
    except:
        print("dataset folder was already erased or not existing")

    print(commitID, file)    
    p = subprocess.Popen('git checkout {0} {1}.dvc'.format(commitID, file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    print("Downloading data...")
    p = subprocess.Popen('dvc pull {0}.dvc'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)

    # check the download
    successful_download = os.path.isdir(path + '\{0}'.format(file))
    while not successful_download:
        print("retrying files download...")
        p = subprocess.Popen('dvc pull {0}.dvc'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        printResponse(p)
        successful_download = os.path.isdir(path + '\{0}'.format(file))

def handleCheckout_Linux(commitID, file):
    try:
        shutil.rmtree(path + '/{0}'.format(file))
    except:
        print("dataset folder was already erased or not existing")
            
    p = subprocess.Popen('git checkout {0} {1}.dvc'.format(commitID, file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    print("Downloading data...")
    p = subprocess.Popen('dvc pull {0}.dvc'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)

    # check the download
    successful_download = os. path. isdir(path + '/{0}'.format(file))
    while not successful_download:
        print("retrying files download...")
        p = subprocess.Popen('dvc pull {0}.dvc'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        printResponse(p)
        successful_download = os. path. isdir(path + '/{0}'.format(file))
        



def processMetada(Version, file, partitions):
    PATH_TO_FILE = path + "/{0}/dataset_labels.csv".format(file)
    if not os.path.exists(PATH_TO_FILE):
        print("Can't process the metadata file because of missing file 'dataset_labels.csv'")
    else:
        data = pd.read_csv(PATH_TO_FILE)
        TotalSeconds = data["length"].sum()
        TotalMinutes = TotalSeconds / 60
        TotalAudios = len(data["id_audio"].unique()) 

        # average amount of clips by agent
        total_clips_agent = []
        for audio in data["id_audio"].unique():
            total_clips_agent.append(len(data[(data.id_audio == audio) & (data.speaker == "agent")]))

        AVG_clips_agent = sum(total_clips_agent)/len(total_clips_agent)

        # average amount of clips by client
        total_clips_agent = []
        for audio in data["id_audio"].unique():
            total_clips_agent.append(len(data[(data.id_audio == audio) & (data.speaker == "client")]))

        AVG_clips_client = sum(total_clips_agent)/len(total_clips_agent)

        #----------------------------------------------------------------------------------------------------
        # average lenght of clips by agent
        total_clips_agent = []
        for audio in data["id_audio"].unique():
            total_clips_agent.append(sum(data[(data.id_audio == audio) & (data.speaker == "agent")]["length"].astype('int32')) / len(data[(data.id_audio == audio) & (data.speaker == "agent")]["length"]))

        AVG_lenght_agent = sum(total_clips_agent)/len(total_clips_agent)

        # average lenght of clips by client
        total_clips_agent = []
        for audio in data["id_audio"].unique():
            total_clips_agent.append(sum(data[(data.id_audio == audio) & (data.speaker == "client")]["length"].astype('int32')) / len(data[(data.id_audio == audio) & (data.speaker == "client")]["length"]))

        AVG_lenght_client = sum(total_clips_agent)/len(total_clips_agent)

        metadata = {
                "dataset" : Version,
                "TotalAudios" : TotalAudios,
                "TotalSeconds" : TotalSeconds,
                "TotalMinutes" : TotalMinutes,
                "AVG_ClipsByAgent" : AVG_clips_agent,
                "AVG_ClipsByClient" : AVG_clips_client,
                "AVG_LenghtByAgent" : AVG_lenght_agent,
                "AVG_LenghtByClient" : AVG_lenght_client
            }


        if partitions:
            # Train partition data
            data = pd.read_csv(path + "/{0}/train.csv".format(file))
            TrainTotalSeconds = data["length"].sum()
            TrainTotalMinutes = TrainTotalSeconds / 60
            TrainTotalClips = int(data["id_audio"].count())

            # validation partition data
            data = pd.read_csv(path + "/{0}/train.csv".format(file))
            ValTotalSeconds = data["length"].sum()
            ValTotalMinutes = ValTotalSeconds / 60
            ValTotalClips = int(data["id_audio"].count())

            metadata["TrainTotalSeconds"] = TrainTotalSeconds
            metadata["TrainTotalMinutes"] = TrainTotalMinutes
            metadata["TrainTotalClips"] = TrainTotalClips
            metadata["ValTotalSeconds"] = ValTotalSeconds
            metadata["ValTotalMinutes"] = ValTotalMinutes
            metadata["ValTotalClips"] = ValTotalClips


        return metadata

def updateVersion(version):
    number = version.split(".")[1]
    number = int(number) + 1
    version = version[:3] + str(number)
    return version

def generateMetadata(metadata, file):
    with open(path + '/{0}/metadata.json'.format(file), 'w') as json_file:
        json.dump(metadata, json_file)
    print("Metadata file generated")

def readMetadata(file):
    f = open(path + '/{0}/metadata.json'.format(file))
    data = json.load(f)
    return data["dataset"]

def updateLabels(file):
    dir_audios = path + "/{0}/audio_files/".format(file)
    destination = path + "/{0}/".format(file)
    status = prepare_dataset(root_directory=dir_audios, overwrite_df=False, dest_directory=destination)
    return status

def Push(version, file): 
    p = subprocess.Popen('git rm -r --cached "{0}"'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    p = subprocess.Popen('dvc add {0}'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    Addcommand = getGitAddCommand(p)
    p = subprocess.Popen(Addcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    p = subprocess.Popen('git add {0}/metadata.json -f'.format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    p = subprocess.Popen('git commit -m "{0} {1}"'.format(file, version), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    print("Uploading files...")
    p = subprocess.Popen('dvc push', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pushFailed = checkPushError(p)

    # retry push
    while pushFailed:
        p = subprocess.Popen('dvc push', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pushFailed = checkPushError(p)
            
    print("Files uploaded")

    p = subprocess.Popen('git push', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    print("Git push completed")


def getVersion(args):
    version = ""
    try:
        # check if the version number whas explicitly passed
        if args[3]:
            version = args[3]
    except:
        print("Commiting a new dataset")
        try:
            version = updateVersion(readMetadata(args[2]))
        except:
            print("Couldn't find the metadata.json file, check that you're attempting to upload on top of the latest version of the dataset or manually input the version of this dataset.")
            version = "n/a"
    return version




def handle_Windows(args):
    # check for new dataset updates
    p = subprocess.Popen('git pull', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)

    if args[1] == "pull":
        if args[3] == "latest":
            p = subprocess.Popen('git log --oneline', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commitID = getCommitID(p, args[2], "latest")
            handleCheckout(commitID, args[2])
            print("Files ready")

        else:       
            p = subprocess.Popen('git log --oneline', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commitID = getCommitID(p, args[2], args[3])

            if commitID == 0:
                print("Dataset version not found")
            else:
                handleCheckout(commitID, args[2])
                print("Files ready")

    if args[1] == "push":

        print("Should Train-Val partitions be generated?(Y/N)")
        partitions_resp = str(input()).lower()
        calculate_partitions = False
        if partitions_resp == "y":
            calculate_partitions = True

        version = getVersion(args)

        # if the version number can't be defined the process shouldn't continue
        if version != "n/a":
        

            #update dataset labels
            labels_ready = updateLabels(args[2])
            
            if not labels_ready:
                print("Can't continue the execution")
            else:

                # generate Train Val partitions
                if calculate_partitions:
                    generateTrainValPartitions(args[2])

                # update metadata
                metadata = processMetada(version, args[2], calculate_partitions)
                generateMetadata(metadata, args[2])


                # push new dataset changes
                Push(version, args[2])




def handle_Linux(args):
    # check for new dataset updates
    p = subprocess.Popen('git pull', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    printResponse(p)
    
    if args[1] == "pull":
        if args[3] == "latest":
            p = subprocess.Popen('git log --oneline', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commitID = getCommitID(p, args[2], "latest")
            handleCheckout_Linux(commitID, args[2])
            print("Files ready")

        else:       
            p = subprocess.Popen('git log --oneline', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commitID = getCommitID(p, args[2], args[3])

            if commitID == 0:
                print("Dataset version not found")
            else:
                handleCheckout_Linux(commitID, args[2])
                print("Files ready")


    if args[1] == "push":

        print("Should Train-Val partitions be generated?(Y/N)")
        partitions_resp = str(input()).lower()
        calculate_partitions = False
        if partitions_resp == "y":
            calculate_partitions = True

        version = getVersion(args)

        # if the version number can't be defined the process shouldn't continue
        if version != "n/a":
        

            #update dataset labels
            labels_ready = updateLabels(args[2])
            
            if not labels_ready:
                print("Can't continue the execution")
            else:

                # generate Train Val partitions
                if calculate_partitions:
                    generateTrainValPartitions(args[2])

                # update metadata
                metadata = processMetada(version, args[2])
                generateMetadata(metadata, args[2])


                # push new dataset changes
                Push(version, args[2])


