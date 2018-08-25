import imageio

reader = imageio.get_reader('imageio:cockatoo.mp4')
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer('~/cockatoo_gray.mp4', fps=fps)

for im in reader:
    writer.append_data(im[:, :, 1])
writer.close()