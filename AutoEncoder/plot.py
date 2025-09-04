import numpy as np
import matplotlib.pyplot as plt
import torch
from add_noise import add_noise

def plot_ae_outputs_den(encoder, decoder, test_dataset, epoch, n=10, noise_factor=0.3, device='cpu'):
	plt.figure(figsize=(16,4.5))
	targets = test_dataset.targets.numpy()
	t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
	for i in range(n):

		ax = plt.subplot(3,n,i+1)
		img = test_dataset[t_idx[i]][0].unsqueeze(0)
		image_noisy = add_noise(img,noise_factor)
		image_noisy = image_noisy.to(device)

		encoder.eval()
		decoder.eval()

		with torch.no_grad():
			rec_img = decoder(encoder(image_noisy))

		plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if i == n//2:
			ax.set_title(f'Original images, epoch {epoch}')
		ax = plt.subplot(3, n, i + 1 + n)
		plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if i == n//2:
			ax.set_title(f'Corrupted images, epoch {epoch}')

		ax = plt.subplot(3, n, i + 1 + n + n)
		plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if i == n//2:
			ax.set_title(f'Reconstructed images, epoch {epoch}')
	plt.subplots_adjust(left=0.1,
					bottom=0.1,
					right=0.7,
					top=0.9,
					wspace=0.3,
					hspace=0.3)
	plt.savefig(f'results/sample-epoch{epoch}.png')