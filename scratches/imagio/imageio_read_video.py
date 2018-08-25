import imageio
filename = 'data/video/cart/cartpole1.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

for image in vid.iter_data():
    print(image.mean())

metadata = vid.get_meta_data()
print(metadata)