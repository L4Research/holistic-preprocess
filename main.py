import os

from prepare_single import preprocessVideo

BASE_PATH = '/content'
DATA_PATH = '/content/drive/MyDrive/Final_fyp_trims/data'
PREPROCESS_DATA_PATH = '/content/data-holistic/videos'
TRAIN_DATA_PATH = '/content/drive/MyDrive/data-holistic/videos/bg_train_data'
TEST_DATA_PATH = '/content/drive/MyDrive/data-holistic/videos/bg_test_data'

FROM = '002_003_001.mp4'
TO = '005_009_001.mp5'


def preprocess():
    start = False
    for root, dirs, files in sorted(os.walk(DATA_PATH)):
        files.sort()
        for file in files:

            originalVideoPath = DATA_PATH + "/" + file

            if file == FROM:
                start = True

            if(file == TO):
                print(
                    'finished processing files ----------------------------------------')
                return

            print('CURRENT FILE : ', file)

            if start == True:

                print('processing : ', file)

                splits = file.split(".")[0].split('_')
                signId = splits[0]
                personId = splits[1]
                videoId = splits[2]

                print('signId', signId, 'personId',
                      personId, 'videoId', videoId)
                print('-----------------------------------------------')

                intSignId = int(signId)
                if intSignId < 10:
                    signId = '0'+str(intSignId)

                directory = TRAIN_DATA_PATH + "/" + signId
                if not os.path.exists(directory):
                    os.makedirs(directory)

                videoWritePath = directory
                if int(videoId) == 4 or int(videoId) == 5:
                    videoWritePath = TEST_DATA_PATH

                videoWritePath = videoWritePath + "/" + file

                if os.path.exists(videoWritePath):
                    print('file exists in the location')

                else:
                    preprocessVideo(originalVideoPath, videoWritePath)

            # intPersonId = int(personId)
            # directory = directory + "/" + str(intPersonId)
            # if not os.path.exists(directory):
            #   os.makedirs(directory)

            # videoPath = BASE_PATH + DATA_PATH + '/' + file
preprocess()