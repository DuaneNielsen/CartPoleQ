import imageio, os

path = 'data/images/fullscreen/'

fileList = []
for file in os.listdir(path):
    if file.startswith('raw'):
        complete_path = path + file
        fileList.append(complete_path)

writer = imageio.get_writer('test.mp4')

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()